"""Consistency of PlurPO's judgments (appendix "How consistent are the
judgments of PlurPO?"): same-prompt vs cross-prompt label agreement.

Pipeline for one env (slurm/run_reliability.sh runs it end to end):
  1. reliability.py select   250 seeded prompt units from the frozen TEST split
                             (AITA-NTA-FLIP: drawn from the pooled 2000
                             ::orig / ::flip units, so both sides of a pair are
                             not guaranteed to appear) -> [[id, text], ...] json
  2. candidate sampler       10 samples per prompt from the PlurPO adapter,
                             temperature 1.0, top_p 0.95, 1536 max tokens
                             (scripts/train/role_confusion_sample.py --prompts-file)
  3. label_candidates.py     --judge endorse (OEQ/PAS) or verdict (AITA/FLIP)
  4. reliability.py stats    per-prompt counts and the statistics below

For prompt i with n_i responses in the metric's two definite categories
(OEQ/PAS: label 0 challenge or 1 explicit affirm; AITA/FLIP: YTA or NTA) and
y_i in the endorsing one (label 1 / NTA), p_i = y_i / n_i. Prompts with n_i = 0
are excluded and counted.
  C_within  = mean_i[p_i^2 + (1 - p_i)^2]
  p_bar     = mean_i[p_i]      (unweighted, so the identity below holds)
  C_between = p_bar^2 + (1 - p_bar)^2
  gap       = C_within - C_between = 2 Var(p_i)   (asserted)
Outputs: <out-dir>/per_prompt_counts.csv, <out-dir>/reliability_stats.json.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from plurpo.data import ENVS, keyed_prompts, load_split
from plurpo.io import read_jsonl

OEQ_PAS_CATEGORY = {0: "challenges", 1: "explicit_affirm", 2: "implicit_affirm", 3: "neutral"}


def select_units(env, n, seed):
    """Seeded sample of `n` (id, text) units from the env's frozen test split."""
    pool = keyed_prompts(env, load_split(env, "test"))
    return random.Random(seed).sample(pool, n)


def per_prompt_counts(env, cand_df, labels_df):
    """Per-prompt category counts plus n_sampled, y_i, n_i."""
    labels_df = labels_df.copy()
    if env in ("OEQ", "PAS"):
        labels_df["category"] = labels_df["label"].apply(
            lambda v: "unparsed" if pd.isna(v) else OEQ_PAS_CATEGORY.get(int(float(v)), "unparsed"))
        endorse_cat, critical_cat = "explicit_affirm", "challenges"
    else:
        labels_df["category"] = labels_df["label"].fillna("OTHER")
        endorse_cat, critical_cat = "NTA", "YTA"
    n_sampled = cand_df.groupby("id").size().rename("n_sampled")
    counts = labels_df.pivot_table(index="id", columns="category", aggfunc="size", fill_value=0)
    counts = counts.join(n_sampled, how="outer").fillna(0)
    counts["n_sampled"] = counts["n_sampled"].astype(int)
    for c in (endorse_cat, critical_cat):
        if c not in counts.columns:
            counts[c] = 0
    counts["y_i"] = counts[endorse_cat].astype(int)
    counts["n_i"] = (counts[endorse_cat] + counts[critical_cat]).astype(int)
    return counts


def reliability_stats(counts):
    """C_within, C_between and their gap over prompts with n_i > 0."""
    included = counts[counts["n_i"] > 0]
    assert len(included) > 1, "need >1 prompt with n_i>0 to estimate Var(p_i)"
    p = included["y_i"] / included["n_i"]
    p_bar = p.mean()
    c_within = (p ** 2 + (1 - p) ** 2).mean()
    c_between = p_bar ** 2 + (1 - p_bar) ** 2
    var_p = (p ** 2).mean() - p_bar ** 2
    gap = c_within - c_between
    assert abs(gap - 2 * var_p) < 1e-9, f"identity check failed: {gap} != {2 * var_p}"
    return {"n_prompts_included": int(len(included)),
            "n_prompts_excluded_zero_denom": int((counts["n_i"] == 0).sum()),
            "mean_n_i": float(included["n_i"].mean()), "p_bar": float(p_bar),
            "C_within": float(c_within), "C_between": float(c_between),
            "gap_points": float(gap * 100), "var_p": float(var_p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("select", "stats"))
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.step == "select":
        units = select_units(args.env, args.n, args.seed)
        assert len({i for i, _ in units}) == args.n, "duplicate ids in selection"
        out = os.path.join(args.out_dir, f"prompts_{args.n}.json")
        with open(out, "w") as f:
            json.dump(units, f)
        print(f"[reliability] {args.env}: {len(units)} units -> {out}")
        return
    cand = pd.DataFrame(read_jsonl(os.path.join(args.out_dir, "pairs", "candidates.jsonl")))
    cand["id"] = cand["id"].astype(str)
    labels = pd.read_csv(os.path.join(args.out_dir, "pairs", "labels.csv"), dtype={"id": str})
    counts = per_prompt_counts(args.env, cand, labels)
    stats = reliability_stats(counts)
    counts.to_csv(os.path.join(args.out_dir, "per_prompt_counts.csv"))
    with open(os.path.join(args.out_dir, "reliability_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
