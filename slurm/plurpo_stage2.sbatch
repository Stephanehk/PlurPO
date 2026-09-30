#!/bin/bash
# PlurPO stage 2 (role-confusion mitigation): sample -> label -> pairs -> one
# offline RPO epoch continuing the stage-1 adapter. 1 GPU (2 for Qwen3-32B).
#   sbatch slurm/plurpo_stage2.sbatch <config> <env> <stage1_adapter> <output_dir>
#SBATCH --job-name=plurpo_s2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source "$(dirname "$0")/common.sh" 2>/dev/null || source slurm/common.sh
"$PLURPO_PYTHON" scripts/train/run_role_confusion.py --config "$1" --env "$2" \
    --adapter "$3" --out-dir "$4"
