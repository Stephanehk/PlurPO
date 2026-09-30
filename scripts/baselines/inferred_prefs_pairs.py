"""Inferred-Prefs-DPO, pair construction (App. "Inferred-Prefs-DPO").

For each training prompt, the paired human response and 10 sampled Qwen3-8B
candidates each carry a v5 action-endorsement label (0 challenge, 1 explicit
endorse, 2 implicit endorse, 3 neutral; scripts/eval/label_candidates.py).

    chosen   = a candidate whose label EXACTLY matches the human response's
               label for that prompt, drawn uniformly at random (seeded)
    rejected = each candidate whose label differs, at most
               --max-pairs-per-prompt per prompt
one pair per rejected candidate. Truncated candidates (finished=False) and
unparsed labels are dropped. PAS has no human responses, so its human label is
assumed to be 0 (challenge) for every prompt (--assume-human-label 0).

Output: <out-dir>/pairs/pairs.jsonl ({"prompt","chosen","rejected"}) and
manifest.json. Deterministic given its inputs and --seed; reproduces the
paper's pair files byte-for-byte (tests/test_baselines_inferred_prefs.py).
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.io import read_keyed_candidates, write_jsonl

csv.field_size_limit(10**9)


def _label(v):
    """CSV cell -> int label, or None for an empty / NaN / NA cell."""
    if v in ("", "nan", "NaN", "<NA>", None):
        return None
    return int(float(v))


def read_candidate_labels(path):
    """labels.csv (id, sample_idx, label) -> {(id, sample_idx): int or None}."""
    with open(path) as f:
        return {(str(r["id"]), int(r["sample_idx"])): _label(r["label"])
                for r in csv.DictReader(f)}


def read_human_labels(path):
    """Human-label csv (id, ..., endorse_response) -> {id: int or None}."""
    with open(path) as f:
        return {str(r["id"]): _label(r.get("endorse_response", "")) for r in csv.DictReader(f)}


def inferred_prefs_pairs(candidates, labels, human, max_pairs_per_prompt, seed,
                         require_finished=True):
    """Pure pair builder. `candidates`: {(id, sample_idx): row} in file order
    (last write wins); `labels`: {(id, sample_idx): label}; `human`:
    {id: label}. Returns (pairs, manifest); prompts are visited in sorted-id
    order and one seeded RNG drives every draw."""
    by_prompt = {}
    for key, row in candidates.items():
        pid = key[0]
        g = by_prompt.setdefault(pid, {"sentence": row["sentence"], "chosen": [],
                                       "rejected": [], "n_seen": 0, "n_trunc": 0})
        g["n_seen"] += 1
        if require_finished and not row.get("finished", False):
            g["n_trunc"] += 1
            continue
        lab = labels.get(key)
        hlab = human.get(pid)
        if lab is None or hlab is None:
            continue
        (g["chosen"] if lab == hlab else g["rejected"]).append(row["response"])

    rng = random.Random(seed)
    pairs, clen, rlen = [], [], []
    n_no_human = n_no_rejected = n_no_chosen = n_capped = 0
    for pid in sorted(by_prompt):
        g = by_prompt[pid]
        if human.get(pid) is None:
            n_no_human += 1
            continue
        if not g["rejected"]:
            n_no_rejected += 1
            continue
        if not g["chosen"]:
            n_no_chosen += 1
            continue
        rejected = g["rejected"]
        if len(rejected) > max_pairs_per_prompt:
            rejected = rng.sample(rejected, max_pairs_per_prompt)
            n_capped += 1
        for rej in rejected:
            ch = rng.choice(g["chosen"])
            pairs.append({"prompt": g["sentence"], "chosen": ch, "rejected": rej})
            clen.append(len(ch))
            rlen.append(len(rej))
    rng.shuffle(pairs)

    def stats(xs):
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean_chars": round(statistics.mean(xs), 1),
                "median_chars": statistics.median(xs)}

    manifest = {
        "n_candidates": len(candidates),
        "n_prompts_seen": len(by_prompt),
        "n_prompts_with_human_label": len(by_prompt) - n_no_human,
        "n_prompts_contributing": len(by_prompt) - n_no_human - n_no_rejected - n_no_chosen,
        "n_prompts_dropped_no_human_label": n_no_human,
        "n_prompts_dropped_no_match": n_no_chosen,
        "n_prompts_dropped_no_mismatch": n_no_rejected,
        "n_prompts_hit_pair_cap": n_capped,
        "n_truncated_dropped": sum(g["n_trunc"] for g in by_prompt.values()),
        "max_pairs_per_prompt": max_pairs_per_prompt,
        "require_finished": bool(require_finished),
        "seed": seed,
        "n_pairs": len(pairs),
        "chosen_length": stats(clen),
        "rejected_length": stats(rlen),
    }
    return pairs, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="dir holding pairs/candidates.jsonl + labels.csv")
    ap.add_argument("--human-labels", default=None,
                    help="OEQ: label_candidates.py --mode human output csv")
    ap.add_argument("--assume-human-label", type=int, default=None,
                    help="PAS: use this label (0) as every prompt's human label")
    ap.add_argument("--max-pairs-per-prompt", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert (args.human_labels is None) != (args.assume_human_label is None), (
        "pass exactly one of --human-labels / --assume-human-label")
    pdir = os.path.join(args.out_dir, "pairs")
    candidates = read_keyed_candidates(os.path.join(pdir, "candidates.jsonl"))
    labels = read_candidate_labels(os.path.join(pdir, "labels.csv"))
    if args.human_labels:
        human = read_human_labels(args.human_labels)
    else:
        human = {pid: args.assume_human_label for pid, _ in candidates}
    pairs, manifest = inferred_prefs_pairs(candidates, labels, human,
                                           args.max_pairs_per_prompt, args.seed)
    assert pairs, "0 pairs built"
    write_jsonl(os.path.join(pdir, "pairs.jsonl"), pairs)
    with open(os.path.join(pdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
