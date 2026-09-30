#!/bin/bash
# Qwen3-32B stage 1: fresh LoRA + one offline RPO epoch on the Qwen3-8B PlurPO
# pairs (data/plurpo_qwen3-8b_pairs). 2 GPUs (AITA/FLIP shard with device_map=auto).
#   sbatch slurm/qwen3-32b_stage1.sbatch <env> <output_dir>
#SBATCH --job-name=plurpo_32b_s1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
source "$(dirname "$0")/common.sh" 2>/dev/null || source slurm/common.sh
"$PLURPO_PYTHON" scripts/train/train_qwen3_32b_stage1.py --env "$1" --out-dir "$2"
