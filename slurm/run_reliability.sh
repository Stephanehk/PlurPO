#!/bin/bash
# Same-prompt vs cross-prompt agreement for one env (appendix, PlurPO
# consistency table): 250 test prompts x 10 samples from the PlurPO adapter.
#
#   bash slurm/run_reliability.sh STAGE ENV ADAPTER OUT_DIR
#
# STAGE in order: select (CPU) | sample (GPU, sbatch) | label (network) | stats (CPU).
# The sample stage uses the training-side candidate sampler
# (scripts/train/role_confusion_sample.py) with --prompts-file.
set -euo pipefail
STAGE="${1:?}"; ENV="${2:?}"; ADAPTER="${3:?}"; OUT="${4:?}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
case "$STAGE" in
  select) "$PY" "$REPO/scripts/eval/reliability.py" select --env "$ENV" --out-dir "$OUT" --n 250 --seed 0 ;;
  sample)
    # shellcheck disable=SC2086
    sbatch $SBATCH_ARGS --gpus=1 --time=12:00:00 --wrap \
      "$PY $REPO/scripts/train/role_confusion_sample.py --env $ENV --adapter $ADAPTER --out-dir $OUT \
       --prompts-file $OUT/prompts_250.json --n 10 --temperature 1.0 --top-p 0.95 \
       --max-tokens 1536 --seed 0" ;;
  label)
    J=endorse; [[ "$ENV" == AITA || "$ENV" == AITA-NTA-FLIP ]] && J=verdict
    "$PY" "$REPO/scripts/eval/label_candidates.py" --mode candidates --judge "$J" --out-dir "$OUT" \
        --max-cost 10 ;;
  stats) "$PY" "$REPO/scripts/eval/reliability.py" stats --env "$ENV" --out-dir "$OUT" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
