#!/bin/bash
# Inferred-Prefs-DPO baseline (App. "Inferred-Prefs-DPO"), Qwen3-8B, OEQ or PAS.
#
#   bash slurm/run_inferred_prefs.sh STAGE ENV [RUNS_DIR] [ALPHA] [EPOCHS]
#
# STAGE, in order (wait for each stage's GPU job before the next):
#   human   OEQ only: v5-label the human response of every final-train prompt (API)
#   sample  10 candidates per final-train prompt from the UNTRAINED base model (GPU)
#   label   v5-label every candidate (API)
#   pairs   exact-label-match pairs vs the human label (OEQ) / assumed 0 (PAS) (CPU)
#   train   fresh LoRA + offline DPO on the one shared pair file (GPU)
#   gen     attempt-1 held-out generation of the trained adapter (GPU, 2 shards)
#   score   v5 action-endorsement judging of that generation (API)
#
# The paper's reported arm is ALPHA=0 (plain DPO), EPOCHS=3: three passes over ONE
# pair set built once from base-model samples (not three on-policy rounds);
# effective batch 8 (1 GPU, grad_accum 8; e.g. OEQ 1639 pairs -> 3 x 205 steps).
# The alpha 0.25 / 0.5 / 0.6 variants the appendix mentions were EPOCHS=1.
# Generation and scoring are UNGATED (no rejection sampling), as in the paper's
# Inferred-Prefs table. Both train and eval use the FINAL split (the research
# runs are tagged "origdata" but read final_training_eval_data ids).
set -euo pipefail
STAGE="${1:?stage}"; ENV="${2:?OEQ|PAS}"; RUNS="${3:-runs}"; ALPHA="${4:-0}"; EPOCHS="${5:-3}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
PY="${PY:-${PLURPO_PYTHON:-python}}"; SBATCH_ARGS="${SBATCH_ARGS:-}"
LOWER=$(echo "$ENV" | tr 'A-Z' 'a-z')
SHARED="$RUNS/inferred_prefs_${LOWER}_shared"
OUT="$RUNS/inferred_prefs_${LOWER}_rpo${ALPHA}_epoch${EPOCHS}"
HUMAN="$SHARED/human_labels_train.csv"
TAG=final_finaldata_temp1.0_seed0
mkdir -p "$SHARED"
case "$STAGE" in
  human)
    [[ "$ENV" == OEQ ]] || { echo "PAS has no human responses (label 0 is assumed)"; exit 0; }
    "$PY" scripts/eval/label_candidates.py --mode human --env OEQ --split train --out "$HUMAN" ;;
  sample)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS slurm/baselines_gpu.sbatch scripts/train/role_confusion_sample.py \
        --env "$ENV" --out-dir "$SHARED" --n 10 --temperature 1.0 --top-p 0.95 \
        --max-tokens 1536 --seed 0 ;;
  label)
    "$PY" scripts/eval/label_candidates.py --mode candidates --judge endorse --out-dir "$SHARED" ;;
  pairs)
    if [[ "$ENV" == OEQ ]]; then HL=(--human-labels "$HUMAN"); else HL=(--assume-human-label 0); fi
    "$PY" scripts/baselines/inferred_prefs_pairs.py --out-dir "$SHARED" "${HL[@]}" \
        --max-pairs-per-prompt 10 --seed 0 ;;
  train)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS --wrap="set -e; cd $PWD; \
      $PY scripts/train/make_fresh_lora.py --model-id Qwen/Qwen3-8B --out-dir ${OUT}_init; \
      $PY scripts/train/train_offline.py --adapter ${OUT}_init --out-dir $OUT \
          --pairs-file $SHARED/pairs/pairs.jsonl --model-id Qwen/Qwen3-8B --beta 0.1 \
          --rpo-alpha $ALPHA --num-train-epochs $EPOCHS --grad-accum 8 --seed 0" \
      --gres=gpu:1 --mem=200G --time=1-00:00:00 --output=logs/inferred_prefs_train_%j.out ;;
  gen)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS --array=0-1 \
      --export=ALL,GEN_ENV="$ENV",GEN_RUN_DIR="$OUT",GEN_ARGS="",PY="$PY" slurm/eval_generate.sbatch ;;
  score)
    "$PY" scripts/eval/judge_endorsement.py --source attempt1 --run-dir "$OUT" --tag "$TAG" \
        --expected-shards 2 --expect 1000 --name "${LOWER}_inferredprefs_rpo${ALPHA}_epoch${EPOCHS}_finaldata" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
