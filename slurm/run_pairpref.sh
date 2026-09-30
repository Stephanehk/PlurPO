#!/bin/bash
# PlurPO-no-stakeholder-preferences ablation (App. "On the benefits of
# explicitly simulating stakeholders").
#
#   bash slurm/run_pairpref.sh STAGE ENV [RUNS_DIR]      STAGE = train | gen | score
#
# Identical to PlurPO (same stakeholder panel and 2/2/6 candidate mix) except the
# veto stage is skipped: the FROZEN Qwen3-8B labels up to 10 candidate pairs per
# prompt with an A/B preference (PREFERENCE_PROMPT), A/B order alternated, in a
# fixed offset-major order (adjacent candidates first) -- not random pairs.
# Then the standard role-confusion stage. Effective batch 8 (grad_accum 8,
# after the batch-size fix), vs 16 for PlurPO itself: a confound of the
# ablation as run, reproduced here with --grad-accum 8.
# The ablation table uses UNGATED attempt-1 generations.
set -euo pipefail
STAGE="${1:?train|gen|score}"; ENV="${2:?OEQ|PAS}"; RUNS="${3:-runs}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
PY="${PY:-${PLURPO_PYTHON:-python}}"; SBATCH_ARGS="${SBATCH_ARGS:-}"
LOWER=$(echo "$ENV" | tr 'A-Z' 'a-z')
S1="$RUNS/pairpref_qwen3-8b_${LOWER}_stage1"; S2="$RUNS/pairpref_qwen3-8b_${LOWER}"
TAG=final_finaldata_temp1.0_seed0
case "$STAGE" in
  train)
    # shellcheck disable=SC2086
    J1=$(sbatch --parsable $SBATCH_ARGS slurm/plurpo_stage1.sbatch qwen3-8b "$ENV" "$S1" \
        --pair-source preference --preference-max-pairs 10 --grad-accum 8)
    # shellcheck disable=SC2086
    J2=$(sbatch --parsable $SBATCH_ARGS --dependency=afterok:$J1 slurm/plurpo_stage2.sbatch \
        qwen3-8b "$ENV" "$S1" "$S2")
    echo "stage1=$J1 -> $S1"; echo "stage2=$J2 -> $S2" ;;
  gen)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS --array=0-1 \
      --export=ALL,GEN_ENV="$ENV",GEN_RUN_DIR="$S2",GEN_ARGS="",PY="$PY" slurm/eval_generate.sbatch ;;
  score)
    "$PY" scripts/eval/judge_endorsement.py --source attempt1 --run-dir "$S2" --tag "$TAG" \
        --expected-shards 2 --expect 1000 --name "${LOWER}_pairpref_voice2_finaldata" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
