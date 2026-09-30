#!/bin/bash
# DPO-Neutral baseline (Sec. 4.3; App. "Neutral-DPO Implementation Details").
#
#   bash slurm/run_dpo_neutral.sh ENV [RUNS_DIR]        ENV = OEQ | PAS
#
# Trains with scripts/baselines/train_dpo_neutral.py (2 GPUs: policy + vLLM
# sampler). The gpt-5-mini judge is called INSIDE the training loop, so the
# compute node needs outbound network and OPENROUTER_API_KEY (the research
# runs trained on a cluster with egress for this reason). Evaluate the result
# like any trained arm: bash slurm/run_eval_all.sh <stage> ENV <base_run> <out>.
set -euo pipefail
ENV="${1:?OEQ|PAS}"; RUNS="${2:-runs}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (the neutrality judge runs during training)}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
LOWER=$(echo "$ENV" | tr 'A-Z' 'a-z')
OUT="$RUNS/dpo_neutral_qwen3-8b_${LOWER}"
# shellcheck disable=SC2086
sbatch $SBATCH_ARGS --gres=gpu:2 --time=2-00:00:00 --job-name=dpo_neutral \
    slurm/baselines_gpu.sbatch scripts/baselines/train_dpo_neutral.py --env "$ENV" --output-dir "$OUT"
echo "-> $OUT"
