"""PlurPO stage 2, step 3: labelled candidates -> preference pairs.

Within each prompt, every stakeholder-voiced candidate (label 2) is rejected
against a uniformly drawn assistant-voiced candidate (label 0) of the same
prompt (seeded), at most --max-pairs-per-prompt per prompt; truncated samples
are dropped. See plurpo.pairs.role_confusion_pairs.

Output: <out-dir>/pairs/pairs.jsonl and manifest.json (counts and the
chosen/rejected length balance).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.io import read_keyed_candidates, read_labels_csv, write_jsonl
from plurpo.pairs import role_confusion_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pairs-per-prompt", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pdir = os.path.join(args.out_dir, "pairs")
    pairs, manifest = role_confusion_pairs(
        read_keyed_candidates(os.path.join(pdir, "candidates.jsonl")),
        read_labels_csv(os.path.join(pdir, "labels.csv")),
        args.max_pairs_per_prompt, args.seed)
    assert pairs, f"0 pairs built; label counts {manifest['label_counts']}"
    write_jsonl(os.path.join(pdir, "pairs.jsonl"), pairs)
    with open(os.path.join(pdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
