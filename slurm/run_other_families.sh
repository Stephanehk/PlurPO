#!/bin/bash
# PlurPO for the other model families (App. "Training other models with PlurPO"):
# Phi-4, Llama-3.1-8B-Instruct, Granite-4.1-8B, each on all four environments.
#
#   bash slurm/run_other_families.sh [RUNS_DIR]
#
# These models were trained and sampled in the environment of
# requirements-other-families.txt (newer transformers / vLLM; Granite-4.1
# needs it): point PLURPO_PYTHON at that interpreter before running.
# Per-family knobs (veto prompt, user stakeholder, candidate mix, epochs, LoRA
# targets, vLLM context) live in configs/<family>.json.
set -euo pipefail
RUNS="${1:-runs}"
cd "$(dirname "$0")/.."
: "${PLURPO_PYTHON:?point PLURPO_PYTHON at the requirements-other-families.txt env}"
for CONFIG in phi-4 llama-3.1-8b granite-4.1-8b; do
  for ENV in OEQ PAS AITA AITA-NTA-FLIP; do
    bash slurm/run_plurpo.sh "$CONFIG" "$ENV" "$RUNS"
  done
done
