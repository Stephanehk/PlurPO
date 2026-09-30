#!/bin/bash
# GEPA prompt-optimisation baselines (App. "Evaluating methods that target a
# specific action endorsement rate"), Qwen3-8B, OEQ.
#
#   bash slurm/run_gepa.sh per_response|rate_match [RUNS_DIR]
#
# One GPU (Qwen3-8B on vLLM) plus network: gpt-5-mini is both the reflection LM
# and the metric's judge, so the node needs OPENROUTER_API_KEY and egress.
# per_response: dspy.GEPA auto="heavy" (the paper's ~8000 metric calls, ~2 h).
# rate_match:   gepa.optimize with the rate-matching adapter, patience 10,
#               <= 40 proposals, <= $20 marginal spend (the research settings).
set -euo pipefail
ARM="${1:?per_response|rate_match}"; RUNS="${2:-runs}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
# shellcheck disable=SC2086
sbatch $SBATCH_ARGS --time=08:00:00 --job-name="gepa_$ARM" slurm/baselines_gpu.sbatch \
    "scripts/baselines/gepa/run_${ARM}.py" --out-dir "$RUNS/gepa_$ARM"
