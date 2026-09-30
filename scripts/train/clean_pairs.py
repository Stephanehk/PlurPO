"""Apply the release defect filter to an existing preference-pair file.

Same filter as `rebuild_pairs.py --clean` (drop error-sentinel prompts and
pairs whose prompt matches a held-out TEST prompt of the env after
whitespace/case normalisation), for pair files that are not rebuilt from
sampling records -- e.g. the role-confusion (stage-2) `pairs.jsonl` written by
`plurpo.pairs.role_confusion_pairs`.

Writes `<out>` (kept pairs, original order) and, when `--manifest` is given, a
copy of that manifest extended with a "release_clean" block of removal counts.

Usage:
    python scripts/train/clean_pairs.py --env PAS --in <run>/pairs/pairs.jsonl \
        --manifest <run>/pairs/manifest.json --out data/<dir>/pas_pairs.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plurpo.data import ENVS
from plurpo.io import read_jsonl, write_jsonl
from rebuild_pairs import clean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="source manifest.json to carry over")
    args = ap.parse_args()

    kept, counts = clean(args.env, read_jsonl(args.in_path))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_jsonl(args.out, kept)
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        manifest["release_clean"] = counts
        out_manifest = os.path.splitext(args.out)[0] + "_manifest.json"
        with open(out_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
    print(f"[clean_pairs] wrote {len(kept)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
