#!/bin/bash
# Qwen3-8B + in-context-examples (App. "in-context-examples"), OEQ only.
#
#   bash slurm/run_kshot.sh STAGE [K] [BASE_RUN_DIR]      STAGE = gen | score
#
# Each held-out prompt is preceded by K (post, human response) exemplar turns
# drawn without replacement from data/splits/oeq_train (random.Random(0) per
# shard, advanced in row order -- exactly the research sampler). The paper
# reports K=5 and also ran K=2 and K=10; all three were scored on
# data/splits/oeq_test (not on a separate held-out set). Generation is
# attempt-1 (UNGATED), as in the paper's in-context table.
set -euo pipefail
STAGE="${1:?gen|score}"; K="${2:-5}"; BASE_RUN="${3:-runs/qwen3-8b_base_oeq}"
cd "$(dirname "$0")/.."; mkdir -p logs "$BASE_RUN"
PY="${PY:-${PLURPO_PYTHON:-python}}"; SBATCH_ARGS="${SBATCH_ARGS:-}"
TAG="base-${K}shothuman_finaldata_temp1.0_seed0"
case "$STAGE" in
  gen)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS --array=0-1 \
      --export=ALL,GEN_ENV=OEQ,GEN_RUN_DIR="$BASE_RUN",GEN_ARGS="--base --k-shot-human $K --exemplar-seed 0",PY="$PY" \
      slurm/eval_generate.sbatch ;;
  score)
    "$PY" scripts/eval/judge_endorsement.py --source attempt1 --run-dir "$BASE_RUN" --tag "$TAG" \
        --expected-shards 2 --expect 1000 --name "oeq_kshothuman${K}_finaldata" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
