# Shared environment for every PlurPO job. Edit once for your cluster.
#   PLURPO_PYTHON  interpreter of an env built from requirements.txt
#   HF_HOME        HuggingFace cache holding the base models (compute nodes may
#                  be offline: pre-download with `huggingface-cli download <id>`)
set -euo pipefail
: "${PLURPO_PYTHON:=python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "[plurpo] job=${SLURM_JOB_ID:-local} node=$(hostname) start=$(date -Is)"
