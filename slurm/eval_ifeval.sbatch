#!/bin/bash
# IFEval generation + scoring for one model configuration (one GPU).
#
#SBATCH --job-name=plurpo_ifeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#
# Env: IFE_TAG (output stem, e.g. base / plurpo_oeq), IFE_ADAPTER (empty = base
# model), IFE_MODEL_ID (default Qwen/Qwen3-8B), PY.
set -euo pipefail
: "${IFE_TAG:?}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
SRC=(--base); [[ -n "${IFE_ADAPTER:-}" ]] && SRC=(--adapter "$IFE_ADAPTER")
"$PY" "$REPO/scripts/eval/ifeval_generate.py" "${SRC[@]}" --tag "$IFE_TAG" \
    --model-id "${IFE_MODEL_ID:-Qwen/Qwen3-8B}" --seed 0
"$PY" "$REPO/scripts/eval/ifeval_score.py" --tag "$IFE_TAG" \
    --responses "$REPO/outputs/ifeval/responses/responses_ifeval_${IFE_TAG}_greedy_seed0.jsonl"
