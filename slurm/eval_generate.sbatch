#!/bin/bash
# Held-out generation for ONE arm (one GPU per shard; use a job array over
# shards). Cluster-agnostic template: set the account/partition lines for your
# cluster, or pass them on the sbatch command line.
#
#SBATCH --job-name=plurpo_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# Required env:
#   GEN_ENV       OEQ | PAS | AITA | AITA-NTA-FLIP
#   GEN_RUN_DIR   run dir (outputs go to $GEN_RUN_DIR/eval; default adapter)
#   GEN_ARGS      arm flags for scripts/eval/generate.py, e.g.
#                   "--base"                                  (Qwen3-8B)
#                   "--base --steer becritical"               ("Be critical")
#                   "--base --steer explicit"                 (Cheng et al. 2026)
#                   "--base --steer sharma"                   (Sharma et al. 2024)
#                   "--base --steer dontbesyco"               ("Don't be sycophantic")
#                   "--base --perspective-shift --assess-narrative"
#                   ""                                        (the run dir's adapter)
# Optional env:
#   GEN_MODEL_ID  base model (default Qwen/Qwen3-8B)
#   GEN_EXTRA     further flags, e.g. "--device-map auto" for Qwen3-32B,
#                 "--gate-loop --worklist <json>" for rejection sampling
#   PY            python interpreter (default: python)
# Array: --array=0-1 for OEQ/PAS/FLIP (2 shards), --array=0 for AITA and for
# --gate-loop runs (unsharded).
set -euo pipefail
: "${GEN_ENV:?}"; : "${GEN_RUN_DIR:?}"; : "${GEN_ARGS?set GEN_ARGS (may be empty)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/eval/generate.py" ]] && REPO="$SLURM_SUBMIT_DIR"
PY="${PY:-python}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_ARGS=(--shard-id "$SHARD")
[[ "${GEN_EXTRA:-}" == *--gate-loop* ]] && SHARD_ARGS=()
# shellcheck disable=SC2086
"$PY" "$REPO/scripts/eval/generate.py" \
    --env "$GEN_ENV" --run-dir "$GEN_RUN_DIR" \
    --model-id "${GEN_MODEL_ID:-Qwen/Qwen3-8B}" --seed 0 \
    "${SHARD_ARGS[@]}" $GEN_ARGS ${GEN_EXTRA:-}
