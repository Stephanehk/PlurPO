"""Do stakeholder vetoes merely track action endorsement? (appendix table
P(veto | challenge) and P(veto | endorse) per epoch.)

For each epoch k of a PlurPO stage-1 run, reads the persisted
`<run-dir>/responses_iter_<k>.jsonl` (every candidate with its recorded veto
outcome; the veto is NOT re-run), drops responses whose veto status was
unparsable (`excluded`), samples 500 (prompt, response) rows without
replacement (random.Random(0) per epoch), and labels each with the v5
action-endorsement judge. Labels collapse to challenge = 0 and endorse = {1, 2};
label 3 (neutral) is excluded from both conditionals and counted.

Outputs (--out-dir): iter<k>_labels.csv (key, sentence, response, source,
prompt_idx, vetoed, endorse_response, raw, ...) and summary.json with
P(veto|challenge), P(veto|endorse), their n and Wilson 95% intervals.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from judge_endorsement import score
from plurpo.io import read_jsonl
from plurpo.metrics import wilson


def load_candidates(run_dir, it):
    """(prompt, response) rows of epoch `it` minus excluded responses."""
    out = []
    for row in read_jsonl(os.path.join(run_dir, f"responses_iter_{it}.jsonl")):
        vetoed, excluded = set(row["vetoed"]), set(row.get("excluded", []))
        for pos, (resp, src) in enumerate(zip(row["responses"], row["sources"])):
            if resp in excluded:
                continue
            out.append({"key": f"iter{it}::{row['prompt_idx']}::{pos}", "sentence": row["prompt"],
                        "response": resp, "source": src, "prompt_idx": row["prompt_idx"],
                        "vetoed": resp in vetoed})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="PlurPO stage-1 run dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cost-per-iter", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for it in args.iters:
        cands = load_candidates(args.run_dir, it)
        idx = random.Random(args.seed).sample(range(len(cands)), args.n_sample)
        df = pd.DataFrame([cands[i] for i in idx])
        scored = score(df, "v5", os.path.join(args.out_dir, f"iter{it}_labels.csv"),
                       16, 50, args.max_cost_per_iter)
        label = pd.to_numeric(scored["endorse_response"], errors="coerce")
        ch, en = scored[label == 0], scored[label.isin([1, 2])]
        v_ch, v_en = int(ch["vetoed"].sum()), int(en["vetoed"].sum())
        p_ch, lo_ch, hi_ch = wilson(v_ch, len(ch))
        p_en, lo_en, hi_en = wilson(v_en, len(en))
        rows.append({"iter": it, "n_sampled": len(df),
                     "n_challenge": len(ch), "veto_given_challenge": v_ch,
                     "p_veto_given_challenge": p_ch, "ci_lo_challenge": lo_ch,
                     "ci_hi_challenge": hi_ch,
                     "n_endorse": len(en), "veto_given_endorse": v_en,
                     "p_veto_given_endorse": p_en, "ci_lo_endorse": lo_en, "ci_hi_endorse": hi_en,
                     "n_neutral_dropped": int((label == 3).sum()),
                     "n_unlabeled": int(label.isna().sum())})
        print(f"[iter {it}] P(veto|challenge)={p_ch:.3f} n={len(ch)}  "
              f"P(veto|endorse)={p_en:.3f} n={len(en)}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
