#!/bin/bash
# Generic GPU job for the baseline/ablation drivers: runs one PlurPO script.
#   sbatch [--gres=gpu:2] slurm/baselines_gpu.sbatch <script.py relative to repo> [args...]
# Set PLURPO_PYTHON (or PY) to the env's interpreter and HF_HOME to the model
# cache; add your cluster's account/partition via SBATCH_* env vars or flags.
#SBATCH --job-name=plurpo_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
PY="${PY:-${PLURPO_PYTHON:-python}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
echo "[baseline] job=${SLURM_JOB_ID:-local} node=$(hostname) cmd=$*"
"$PY" "$@"
