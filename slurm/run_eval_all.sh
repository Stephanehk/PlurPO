#!/bin/bash
# Full held-out evaluation for one model on one environment: the base model,
# the six prompting baselines, and one trained adapter (e.g. PlurPO).
#
#   bash slurm/run_eval_all.sh STAGE ENV BASE_RUN_DIR TRAINED_RUN_DIR [MODEL_ID]
#
# STAGE, in order (each is idempotent; wait for the GPU jobs of a stage to
# finish before starting the next):
#   gen       submit attempt-1 generation for every arm       (GPU, sbatch)
#   gate      judge attempt-1 responses, write worklists      (network, CPU)
#   resample  submit in-loop rejection sampling per worklist  (GPU + network)
#   score     endorsement (OEQ/PAS) or verdict (AITA/FLIP) judging of attempt-1
#             responses and of the rescued responses; ELEPHANT validation and
#             framing judges; role-confusion judge on OEQ/PAS (network, CPU)
#
# BASE_RUN_DIR holds the base-model arms (created if missing); TRAINED_RUN_DIR
# is the trained adapter's run dir (its own adapter is evaluated as "final").
# Scores land in outputs/scores/ (see scripts/eval/judge_*.py for layouts);
# scripts/analysis turns them into the paper's tables and figures.
# Needs OPENROUTER_API_KEY for gate/resample/score. Set PY, and SBATCH_ARGS for
# your cluster's account/partition.
set -euo pipefail
STAGE="${1:?stage}"; ENV="${2:?env}"; BASE_RUN="${3:?base run dir}"; TRAINED_RUN="${4:?trained run dir}"
MODEL_ID="${5:-Qwen/Qwen3-8B}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
GEN_EXTRA="${GEN_EXTRA:-}"          # e.g. "--device-map auto" for Qwen3-32B
TAGSUF=_finaldata_temp1.0_seed0
mkdir -p "$BASE_RUN" logs

case "$ENV" in
  OEQ|PAS)       NSHARDS=2 ;;
  AITA)          NSHARDS=1 ;;
  AITA-NTA-FLIP) NSHARDS=2 ;;
  *) echo "unknown env $ENV" >&2; exit 1 ;;
esac
LOWER=$(echo "$ENV" | tr 'A-Z' 'a-z' | sed 's/aita-nta-flip/flip/')

# name | run dir | generate.py arm flags | adapter tag
ARMS=(
  "base|$BASE_RUN|--base|base"
  "becritical|$BASE_RUN|--base --steer becritical|base-becritical"
  "explicit|$BASE_RUN|--base --steer explicit|base-explicit"
  "sharma|$BASE_RUN|--base --steer sharma|base-sharma"
  "dontbesyco|$BASE_RUN|--base --steer dontbesyco|base-dontbesyco"
  "perspshiftassess|$BASE_RUN|--base --perspective-shift --assess-narrative|base-perspshiftassess"
  "stakeholder|$BASE_RUN|--base --steer stakeholder|base-stakeholder"
  "trained|$TRAINED_RUN||final"
)

for spec in "${ARMS[@]}"; do
  IFS='|' read -r NAME RUN FLAGS ATAG <<< "$spec"
  TAG="${ATAG}${TAGSUF}"
  WL="$RUN/eval/gate/resample_${TAG}_a2.json"
  case "$STAGE" in
    gen)
      # shellcheck disable=SC2086
      sbatch $SBATCH_ARGS --array=0-$((NSHARDS - 1)) \
        --export=ALL,GEN_ENV="$ENV",GEN_RUN_DIR="$RUN",GEN_ARGS="$FLAGS",GEN_MODEL_ID="$MODEL_ID",GEN_EXTRA="$GEN_EXTRA",PY="$PY" \
        "$REPO/slurm/eval_generate.sbatch" ;;
    gate)
      "$PY" "$REPO/scripts/eval/gate.py" --run-dir "$RUN" --env "$ENV" --tag "$TAG" \
          --expected-shards "$NSHARDS" ;;
    resample)
      [[ -f "$WL" ]] || { echo "[$NAME] no worklist (every response passed)"; continue; }
      # shellcheck disable=SC2086
      sbatch $SBATCH_ARGS --array=0 \
        --export=ALL,GEN_ENV="$ENV",GEN_RUN_DIR="$RUN",GEN_ARGS="$FLAGS",GEN_MODEL_ID="$MODEL_ID",GEN_EXTRA="$GEN_EXTRA --gate-loop --worklist $WL",PY="$PY" \
        "$REPO/slurm/eval_generate.sbatch" ;;
    score)
      if [[ "$ENV" == OEQ || "$ENV" == PAS ]]; then
        "$PY" "$REPO/scripts/eval/judge_endorsement.py" --source attempt1 --run-dir "$RUN" \
            --tag "$TAG" --expected-shards "$NSHARDS" --expect 1000 --name "${LOWER}_${NAME}_finaldata"
        [[ -f "$WL" ]] && "$PY" "$REPO/scripts/eval/judge_endorsement.py" --source gateloop \
            --run-dir "$RUN" --tag "$TAG" --name "${LOWER}_${NAME}_gateloop"
        "$PY" "$REPO/scripts/eval/judge_role_confusion.py" --run-dir "$RUN" --env "$ENV" \
            --tag "$TAG" --expected-shards "$NSHARDS"
      else
        "$PY" "$REPO/scripts/eval/classify_verdicts.py" --source attempt1 --run-dir "$RUN" \
            --env "$ENV" --tag "$TAG" --num-shards "$NSHARDS"
        [[ -f "$WL" ]] && "$PY" "$REPO/scripts/eval/classify_verdicts.py" --source gateloop \
            --run-dir "$RUN" --env "$ENV" --tag "$TAG"
      fi
      "$PY" "$REPO/scripts/eval/judge_elephant.py" --source attempt1 --env "$ENV" --run-dir "$RUN" \
          --tag "$TAG" --expected-shards "$NSHARDS" --name "$NAME"
      [[ -f "$WL" ]] && "$PY" "$REPO/scripts/eval/judge_elephant.py" --source gateloop --env "$ENV" \
          --run-dir "$RUN" --tag "$TAG" --name "${NAME}_gateloop"
      true ;;
    *) echo "unknown stage $STAGE" >&2; exit 1 ;;
  esac
done

# The OEQ human reference (never gated).
if [[ "$STAGE" == score && "$ENV" == OEQ ]]; then
  "$PY" "$REPO/scripts/eval/judge_endorsement.py" --source human --expect 1000 --name oeq_human_finaldata
fi
