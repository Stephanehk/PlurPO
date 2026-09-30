#!/bin/bash
# Iterative-RPO alpha sweep (App. "Tuning the Iterative RPO alpha parameter").
#
#   bash slurm/run_alpha_sweep.sh STAGE [RUNS_DIR]     STAGE = train | gen | score
#
# PlurPO stage 1 only (no role-confusion stage), OEQ, trained on the TUNING split
# (data/tuning/oeq_train) and evaluated on data/tuning/oeq_test, for
# alpha in {0 (plain DPO), 0.25, 0.5, 0.6, 0.7, 0.8}. Every other setting is
# configs/qwen3-8b.json (3 epochs, P6 + user stakeholder, 2/2/6 candidate mix,
# "Assess this narrative." on the rewrite, effective batch 16 -- the research
# sweep ran on the same pre-fix 2-GPU body as the final runs).
# The table reports the endorsement rate (v5 judge) and the role-confusion rate
# (scripts/eval/judge_role_confusion.py) of UNGATED attempt-1 generations.
#
# `gen` uses generate.py --split-set tuning: the research sweep was generated
# and scored on the tuning test ids.
set -euo pipefail
STAGE="${1:?train|gen|score}"; RUNS="${2:-runs}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
PY="${PY:-${PLURPO_PYTHON:-python}}"; SBATCH_ARGS="${SBATCH_ARGS:-}"
TAG=final_finaldata_temp1.0_seed0
for A in 0 0.25 0.5 0.6 0.7 0.8; do
  OUT="$RUNS/alpha_sweep_oeq_rpo${A}"
  case "$STAGE" in
    train)
      # shellcheck disable=SC2086
      sbatch $SBATCH_ARGS slurm/plurpo_stage1.sbatch qwen3-8b OEQ "$OUT" \
          --split-set tuning --rpo-alpha "$A" ;;
    gen)
      # shellcheck disable=SC2086
      sbatch $SBATCH_ARGS --array=0-1 \
        --export=ALL,GEN_ENV=OEQ,GEN_RUN_DIR="$OUT",GEN_ARGS="--split-set tuning",PY="$PY" \
        slurm/eval_generate.sbatch ;;
    score)
      "$PY" scripts/eval/judge_endorsement.py --source attempt1 --run-dir "$OUT" --tag "$TAG" \
          --expected-shards 2 --expect 1000 --name "oeq_alpha_sweep_rpo${A}_tuning"
      "$PY" scripts/eval/judge_role_confusion.py --run-dir "$OUT" --env OEQ --tag "$TAG" \
          --expected-shards 2 ;;
    *) echo "unknown stage $STAGE" >&2; exit 1 ;;
  esac
done
