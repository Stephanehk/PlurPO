#!/bin/bash
# Qwen3-8B + consider-stakeholders-prompt ablation baseline (App. "On the
# benefits of explicitly simulating stakeholders"): appends
# "consider stakeholders that may be affected by user behavior, and provide an
# objective response" (verbatim, lower-case, no period) to each held-out post.
#
#   bash slurm/run_stakeholder_prompt.sh STAGE ENV [BASE_RUN_DIR]   STAGE = gen | score
#
# UNGATED attempt-1 generation, matching the other rows of that table. (The
# same arm is also part of slurm/run_eval_all.sh, which additionally applies
# rejection sampling.)
set -euo pipefail
STAGE="${1:?gen|score}"; ENV="${2:?OEQ|PAS}"; BASE_RUN="${3:-runs/qwen3-8b_base_$(echo "$2" | tr 'A-Z' 'a-z')}"
cd "$(dirname "$0")/.."; mkdir -p logs "$BASE_RUN"
PY="${PY:-${PLURPO_PYTHON:-python}}"; SBATCH_ARGS="${SBATCH_ARGS:-}"
LOWER=$(echo "$ENV" | tr 'A-Z' 'a-z')
TAG=base-stakeholder_finaldata_temp1.0_seed0
case "$STAGE" in
  gen)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS --array=0-1 \
      --export=ALL,GEN_ENV="$ENV",GEN_RUN_DIR="$BASE_RUN",GEN_ARGS="--base --steer stakeholder",PY="$PY" \
      slurm/eval_generate.sbatch ;;
  score)
    "$PY" scripts/eval/judge_endorsement.py --source attempt1 --run-dir "$BASE_RUN" --tag "$TAG" \
        --expected-shards 2 --expect 1000 --name "${LOWER}_stakeholder_finaldata" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
