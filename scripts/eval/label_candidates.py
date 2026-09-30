"""Label sampled candidates (or human responses) with an evaluation judge.

MODES
  --mode candidates --judge endorse   OEQ/PAS: v5 action-endorsement label 0-3
  --mode candidates --judge verdict   AITA/FLIP: YTA / NTA / UNCLEAR (OTHER if
                                      unparsable)
      input  <out-dir>/pairs/candidates.jsonl  (id, sample_idx, sentence,
             response, finished), as written by the candidate sampler
      output <out-dir>/pairs/labels_raw.csv (all judge columns) and
             <out-dir>/pairs/labels.csv (id, sample_idx, label)
  --mode human --env OEQ              the human response of every prompt in a
                                      split, v5 label, e.g. for the
                                      Inferred-Prefs-DPO baseline
      output --out csv (id, sentence, response, key, label, raw, ...,
             endorse_response)

Used by the reliability analysis (10 samples x 250 prompts) and by the
Inferred-Prefs-DPO baseline. Resumable: rows already labelled are skipped.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from plurpo.data import load_split
from plurpo.io import read_keyed_candidates
from plurpo.judges import endorse_one, verdict_one
from plurpo.llm_api import SpendGuard, make_client

EXTRA = ("raw", "prompt_tokens", "completion_tokens")


def judge_rows(df, judge, out_path, workers, checkpoint, max_cost):
    """Fill df["label"] for unlabelled rows (judge = "endorse" | "verdict"),
    checkpointing to out_path, keyed on df["key"]."""
    for col in ("label",) + EXTRA:
        df[col] = pd.NA
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, dtype={"key": str}).set_index("key")
        for i in df.index:
            if df.at[i, "key"] in prev.index:
                for col in ("label",) + EXTRA:
                    df.at[i, col] = prev.at[df.at[i, "key"], col]
    todo = [i for i in df.index if pd.isna(df.at[i, "label"])]
    print(f"[label] {len(df) - len(todo)}/{len(df)} already labelled; judging {len(todo)}")
    client = make_client()
    guard = SpendGuard(max_cost)

    def call(i):
        if judge == "endorse":
            return endorse_one(client, "v5", df.at[i, "sentence"], df.at[i, "response"])
        return verdict_one(client, df.at[i, "sentence"], df.at[i, "response"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call, i): i for i in todo}
        for count, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            label, r = fut.result()
            if judge == "verdict":
                label = label or "OTHER"
            df.at[i, "label"] = label
            df.at[i, "raw"] = r["content"]
            df.at[i, "prompt_tokens"] = r["prompt_tokens"]
            df.at[i, "completion_tokens"] = r["completion_tokens"]
            if count % checkpoint == 0:
                df.to_csv(out_path, index=False)
            if guard.add(r):
                for f in futs:
                    f.cancel()
                print(f"[label] --max-cost ${max_cost} reached; stopping (resumable)")
                break
    df.to_csv(out_path, index=False)
    print(f"[label] spend ${guard.spent:.3f}; unlabelled {int(df['label'].isna().sum())}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("candidates", "human"))
    ap.add_argument("--judge", choices=("endorse", "verdict"), default=None)
    ap.add_argument("--out-dir", default=None, help="candidates mode")
    ap.add_argument("--env", default="OEQ", choices=("OEQ",), help="human mode")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--split-set", default="final", choices=("final", "tuning"))
    ap.add_argument("--out", default=None, help="human mode output csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--checkpoint", type=int, default=100)
    ap.add_argument("--max-cost", type=float, default=10.0)
    args = ap.parse_args()

    if args.mode == "candidates":
        assert args.out_dir and args.judge, "candidates mode needs --out-dir and --judge"
        rows = read_keyed_candidates(os.path.join(args.out_dir, "pairs", "candidates.jsonl"))
        df = pd.DataFrame([{"id": k[0], "sample_idx": k[1], "sentence": r["sentence"],
                            "response": r["response"]} for k, r in rows.items()])
        df["key"] = df["id"] + "::" + df["sample_idx"].astype(str)
        raw_path = os.path.join(args.out_dir, "pairs", "labels_raw.csv")
        df = judge_rows(df, args.judge, raw_path, args.workers, args.checkpoint, args.max_cost)
        out = os.path.join(args.out_dir, "pairs", "labels.csv")
        df[["id", "sample_idx", "label"]].to_csv(out, index=False)
        print(f"[label] -> {out}")
    else:
        assert args.out, "human mode needs --out"
        split = load_split(args.env, args.split, args.split_set)
        df = pd.DataFrame({"id": split["id"], "sentence": split["post_text"],
                           "response": split["human_response"]})
        df["key"] = df["id"]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df = judge_rows(df, "endorse", args.out, args.workers, args.checkpoint, args.max_cost)
        # `key`/`label` are kept so a rerun resumes; endorse_response is the
        # v5 label under its conventional name.
        df["endorse_response"] = df["label"]
        df.to_csv(args.out, index=False)
        print(f"[label] -> {args.out}")


if __name__ == "__main__":
    main()
