#!/bin/bash
# Submit the full PlurPO training chain for one model config and env:
#   stage 1 -> stage 2 (afterok).     bash slurm/run_plurpo.sh <config> <env> [runs_dir]
# <config> is a file in configs/ (qwen3-8b, phi-4, llama-3.1-8b, granite-4.1-8b).
# Final adapter: <runs_dir>/plurpo_<config>_<env>. Add SBATCH_ACCOUNT /
# SBATCH_PARTITION to the environment for your cluster.
set -euo pipefail
CONFIG="$1"; ENV_NAME="$2"; RUNS="${3:-runs}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
TAG="$(echo "$ENV_NAME" | tr 'A-Z' 'a-z')"
S1="$RUNS/plurpo_${CONFIG}_${TAG}_stage1"; S2="$RUNS/plurpo_${CONFIG}_${TAG}"
J1=$(sbatch --parsable slurm/plurpo_stage1.sbatch "$CONFIG" "$ENV_NAME" "$S1")
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/plurpo_stage2.sbatch "$CONFIG" "$ENV_NAME" "$S1" "$S2")
echo "stage1=$J1 -> $S1"; echo "stage2=$J2 -> $S2"
