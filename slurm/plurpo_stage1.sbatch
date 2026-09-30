#!/bin/bash
# PlurPO stage 1 (iterative stakeholder-veto RPO). Needs 2 GPUs: the trainable
# policy on cuda:0 and the vLLM engine (frozen simulator + policy sampler) on
# cuda:1. ~1 day for Qwen3-8B on 2x H100 (3 epochs, 1000 prompts).
#   sbatch slurm/plurpo_stage1.sbatch <config> <env> <output_dir> [extra args]
#SBATCH --job-name=plurpo_s1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x_%j.out
source "$(dirname "$0")/common.sh" 2>/dev/null || source slurm/common.sh
CONFIG="$1"; ENV_NAME="$2"; OUT="$3"; shift 3
"$PLURPO_PYTHON" scripts/train/train_plurpo.py --config "$CONFIG" --env "$ENV_NAME" \
    --output-dir "$OUT" "$@"
