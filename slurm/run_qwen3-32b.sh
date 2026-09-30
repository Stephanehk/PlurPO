#!/bin/bash
# Qwen3-32B from the Qwen3-8B PlurPO dataset: stage 1 (offline) -> stage 2.
#   bash slurm/run_qwen3-32b.sh <env> [runs_dir]
set -euo pipefail
ENV_NAME="$1"; RUNS="${2:-runs}"
cd "$(dirname "$0")/.."; mkdir -p logs "$RUNS"
TAG="$(echo "$ENV_NAME" | tr 'A-Z' 'a-z')"
S1="$RUNS/plurpo_qwen3-32b_${TAG}_stage1"; S2="$RUNS/plurpo_qwen3-32b_${TAG}"
J1=$(sbatch --parsable slurm/qwen3-32b_stage1.sbatch "$ENV_NAME" "$S1")
J2=$(sbatch --parsable --dependency=afterok:$J1 --gres=gpu:2 slurm/plurpo_stage2.sbatch qwen3-32b "$ENV_NAME" "$S1" "$S2")
echo "stage1=$J1 -> $S1"; echo "stage2=$J2 -> $S2"
