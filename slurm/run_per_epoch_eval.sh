#!/bin/bash
# Per-epoch evaluation of PlurPO stage 1 (App. "Evaluating PlurPO at each epoch").
#
#   bash slurm/run_per_epoch_eval.sh STAGE ENV STAGE1_RUN_DIR     STAGE = gen | score
#
# Generates on data/splits/<env>_test from adapter_iter_1 and adapter_iter_2 of a
# finished stage-1 run (2 shards, attempt-1, UNGATED; tag adapter_iter_<k>),
# reconstructing the research job whose logs are logs/epochgen_4159{39,40,41}*.
# The paper's epoch-3 row is the stage-1 run's FINAL adapter (== adapter_iter_3),
# i.e. before the role-confusion stage -- not the headline PlurPO model; it is
# generated here as tag "final".
set -euo pipefail
STAGE="${1:?gen|score}"; ENV="${2:?OEQ|PAS}"; RUN="${3:?stage-1 run dir}"
cd "$(dirname "$0")/.."; mkdir -p logs
PY="${PY:-${PLURPO_PYTHON:-python}}"; SBATCH_ARGS="${SBATCH_ARGS:-}"
LOWER=$(echo "$ENV" | tr 'A-Z' 'a-z')
for K in 1 2 3; do
  if [[ "$K" == 3 ]]; then ARGS=""; ATAG=final; else ARGS="--adapter $RUN/adapter_iter_$K"; ATAG="adapter_iter_$K"; fi
  TAG="${ATAG}_finaldata_temp1.0_seed0"
  case "$STAGE" in
    gen)
      # shellcheck disable=SC2086
      sbatch $SBATCH_ARGS --array=0-1 \
        --export=ALL,GEN_ENV="$ENV",GEN_RUN_DIR="$RUN",GEN_ARGS="$ARGS",PY="$PY" slurm/eval_generate.sbatch ;;
    score)
      "$PY" scripts/eval/judge_endorsement.py --source attempt1 --run-dir "$RUN" --tag "$TAG" \
          --expected-shards 2 --expect 1000 --name "${LOWER}_p6_epoch${K}_finaldata" ;;
    *) echo "unknown stage $STAGE" >&2; exit 1 ;;
  esac
done
