"""Rebuild (and optionally clean) the stage-1 PlurPO preference pairs of a
finished on-policy run from its persisted sampling records.

The online PlurPO trainer does not save its preference pairs; it saves, per
epoch k, `responses_iter_<k>.jsonl` (one record per training prompt: the 10
sampled responses, the vetoed subset, and the excluded subset). Because the
stakeholder simulator is frozen, those records fully determine the pairs
(`plurpo.pairs.pairs_from_record`). This script re-derives them, checks the
per-epoch counts against the run's `iter_stats.json`, and writes one JSONL with
keys prompt, chosen, rejected, epoch (1-based), prompt_idx.

This is how `data/plurpo_qwen3-8b_pairs/` (the Qwen3-8B pairs used to train
Qwen3-32B) was produced.

Cleaning (`--clean`) removes two known defects of the raw pairs:
  1. error-sentinel prompts: a handful of AITA-NTA-FLIP source rows carry the
     literal prompt "ERROR" (case-insensitive, after strip());
  2. held-out leakage: any pair whose prompt equals a held-out TEST prompt of
     the same env after whitespace/case normalisation (for AITA-NTA-FLIP both
     narrations of every test pair are checked). In PAS one training prompt is
     a whitespace-only variant of a test prompt (pas_01a2c64d vs pas_d92572cf).

Usage:
    python scripts/train/rebuild_pairs.py --env PAS \
        --run-dir runs/dpo_lora_pas_qwen3-8b_..._final \
        --out data/plurpo_qwen3-8b_pairs/pas_pairs.jsonl.gz --clean \
        [--reference runs/reconstructed_final_pairs/pas_pairs_all_epochs.jsonl]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.data import ENVS, load_split
from plurpo.io import read_jsonl, write_jsonl
from plurpo.pairs import pairs_from_record


def normalise(text):
    """Whitespace-collapsed, lower-cased text, used only for leak matching."""
    return " ".join(str(text).split()).lower()


def heldout_prompt_set(env):
    """Normalised texts of every held-out TEST prompt of `env`."""
    df = load_split(env, "test")
    cols = ["post_text", "flipped_post_text"] if env == "AITA-NTA-FLIP" else ["post_text"]
    return {normalise(t) for c in cols for t in df[c]}


def rebuild(run_dir):
    """Rebuild all epochs' pairs of one run.

    Asserts every epoch's pair count equals `iter_stats.json` num_pairs and that
    each records file holds exactly one record per prompt index.
    Returns the list of pair dicts in (epoch, prompt_idx, rejected, chosen) order.
    """
    with open(os.path.join(run_dir, "iter_stats.json")) as f:
        iter_stats = json.load(f)
    out = []
    for stat in iter_stats:
        epoch = int(stat["iter"]) + 1
        records = read_jsonl(os.path.join(run_dir, f"responses_iter_{epoch}.jsonl"))
        idx = [r["prompt_idx"] for r in records]
        assert sorted(idx) == list(range(len(records))), (
            f"epoch {epoch}: prompt_idx is not a permutation of 0..{len(records) - 1}")
        n = 0
        for r in sorted(records, key=lambda r: r["prompt_idx"]):
            for p in pairs_from_record(r):
                out.append(dict(p, epoch=epoch, prompt_idx=r["prompt_idx"]))
                n += 1
        assert n == stat["num_pairs"], (
            f"epoch {epoch}: rebuilt {n} pairs but iter_stats.json says {stat['num_pairs']}")
        print(f"[rebuild] epoch {epoch}: {n} pairs from {len(records)} prompts")
    return out


def check_reference(pairs, reference_path):
    """Assert `pairs` equal a previously reconstructed file (keys prompt,
    chosen, rejected, iter [0-based], prompt_idx) in content and order."""
    ref = read_jsonl(reference_path)
    assert len(ref) == len(pairs), f"reference has {len(ref)} pairs, rebuilt {len(pairs)}"
    for i, (a, b) in enumerate(zip(pairs, ref)):
        same = (a["prompt"] == b["prompt"] and a["chosen"] == b["chosen"]
                and a["rejected"] == b["rejected"] and a["epoch"] == int(b["iter"]) + 1
                and a["prompt_idx"] == b["prompt_idx"])
        assert same, f"pair {i} differs from reference {reference_path}"
    print(f"[rebuild] identical to reference {reference_path}")


def clean(env, pairs):
    """Drop error-sentinel and held-out-leaking pairs (see module docstring).
    Returns (kept, counts) where counts reports what was removed and why."""
    heldout = heldout_prompt_set(env)
    kept = []
    n_error = n_leak = 0
    for p in pairs:
        if p["prompt"].strip().lower() == "error":
            n_error += 1
        elif normalise(p["prompt"]) in heldout:
            n_leak += 1
        else:
            kept.append(p)
    counts = {"n_raw": len(pairs), "n_removed_error_sentinel": n_error,
              "n_removed_heldout_leak": n_leak, "n_clean": len(kept)}
    print(f"[clean] {counts}")
    return kept, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--run-dir", required=True,
                    help="finished stage-1 run dir holding responses_iter_*.jsonl + iter_stats.json")
    ap.add_argument("--out", required=True, help="output pairs jsonl")
    ap.add_argument("--clean", action="store_true", help="apply the defect filter")
    ap.add_argument("--reference", default=None,
                    help="optional previously reconstructed pairs file to verify against")
    args = ap.parse_args()

    pairs = rebuild(args.run_dir)
    if args.reference:
        check_reference(pairs, args.reference)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.clean:
        pairs, counts = clean(args.env, pairs)
        with open(os.path.splitext(args.out.removesuffix(".gz"))[0] + "_clean_counts.json", "w") as f:
            json.dump(counts, f, indent=2)
    write_jsonl(args.out, pairs)
    print(f"[rebuild] wrote {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
