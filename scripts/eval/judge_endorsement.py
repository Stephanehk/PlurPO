"""Action-endorsement judge for OEQ / PAS (gpt-5-mini via OpenRouter).

Labels each (question, response) 0 challenges / 1 explicitly affirms /
2 implicitly affirms / 3 neutral the user's action. The paper's metric is the
action endorsement rate #1 / (#0 + #1). `--prompt-version v5` (default) is the
paper's judge; `v1` is the original Cheng et al. (2025) prompt used for the
appendix comparison.

SOURCES (--source), all read from a generate.py run dir except `human`/`csv`:
  attempt1  every attempt-1 response of one arm (concatenated shards)
  gateloop  the responses the rejection-sampling loop rescued (gate_passed==1
            in eval/gate/responses_<tag>_gateloop.csv)
  gated     the full post-gate scored set in one file: attempt-1 responses the
            gate passed, then the rescued ones (used for the v1 appendix run)
  human     OEQ only: the human responses of the frozen test split
  csv       any csv with `sentence`, `response` (and optionally `id`) columns

The paper's gated rate for an arm = labels of the attempt-1 passers (source
attempt1, filtered by gate_<tag>_a1.csv) + labels of source gateloop; prompts
never rescued are excluded. scripts/analysis does that join.

OUTPUT (--scores-root, default <repo>/outputs/scores):
  inputs/<name>.csv                         the judged (id, sentence, response)
  endorse_gpt5mini_<v>_temp0/scored_<name>.csv            (attempt1/human/csv)
  endorse_gpt5mini_<v>_temp0_gateloop/scored_<name>.csv   (gateloop)
  endorse_gpt5mini_<v>_gated/scored_<name>.csv            (gated)
  columns: id, sentence, response, endorse_response (0-3 or empty), raw,
  prompt_tokens, completion_tokens, reasoning_tokens, served_model.
Resumable: rows whose `raw` already parses to a 0-3 digit are not re-judged.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd

from plurpo.data import load_split
from plurpo.eval_outputs import load_attempt1, load_gateloop
from plurpo.metrics import action_endorsement_rate
from plurpo.judge_parsing import first_endorse_label
from plurpo.judges import endorse_one
from plurpo.llm_api import SpendGuard, make_client
from plurpo.paths import REPO_ROOT

DEFAULT_SCORES_ROOT = os.path.join(REPO_ROOT, "outputs", "scores")
TOMBSTONES = {"[deleted]", "[removed]", "deleted", "removed", ""}
SCORE_COLS = ("endorse_response", "raw", "prompt_tokens", "completion_tokens",
              "reasoning_tokens", "served_model")


def attempt1_frame(run_dir, tag, expected_shards):
    """All attempt-1 OEQ/PAS responses of one arm, shard order."""
    return load_attempt1(run_dir, "OEQ", tag, expected_shards)[["id", "sentence", "response"]]


def gateloop_frame(run_dir, tag):
    """Responses rescued by the gate loop (gate_passed == 1)."""
    s = load_gateloop(run_dir, "OEQ", tag)
    assert s is not None, f"{run_dir}: no gate-loop responses for {tag}"
    s = s[s["gate_passed"].astype(bool)].reset_index(drop=True)
    return s[["id", "sentence", "response"]]


def gated_frame(run_dir, tag, expected_shards):
    """Attempt-1 gate passers followed by rescued responses."""
    a1 = attempt1_frame(run_dir, tag, expected_shards)
    g = pd.read_csv(os.path.join(run_dir, "eval", "gate", f"gate_{tag}_a1.csv"),
                    dtype={"id": str})
    keep = set(g[g["gate"] == 1]["id"])
    a1 = a1[a1["id"].astype(str).isin(keep)]
    frames = [a1]
    if load_gateloop(run_dir, "OEQ", tag) is not None:
        frames.append(gateloop_frame(run_dir, tag))
    return pd.concat(frames, ignore_index=True)


def human_frame():
    """OEQ held-out human responses; refuses tombstoned or empty replies."""
    test = load_split("OEQ", "test")
    resp = test["human_response"].astype(str)
    assert not resp.str.strip().str.lower().isin(TOMBSTONES).any(), (
        "unusable human responses ([deleted]/[removed]/empty)")
    return pd.DataFrame({"id": test["id"], "sentence": test["post_text"], "response": resp})


def score(df, version, out_path, workers, checkpoint, max_cost):
    """Judge every unlabelled row of df, checkpointing to out_path."""
    df = df.reset_index(drop=True).copy()
    for col in SCORE_COLS:
        df[col] = pd.NA
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        assert len(prev) == len(df) and (
            prev["sentence"].astype(str).str.strip() == df["sentence"].astype(str).str.strip()).all(), (
            f"{out_path}: existing file does not match this input; refusing to resume")
        for col in SCORE_COLS:
            df[col] = prev[col]
    todo = [i for i in df.index if first_endorse_label(df.at[i, "raw"]) is None]
    print(f"[endorse] {len(df) - len(todo)}/{len(df)} already judged; judging {len(todo)}")
    client = make_client()
    guard = SpendGuard(max_cost)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(endorse_one, client, version, df.at[i, "sentence"],
                             df.at[i, "response"]): i for i in todo}
        for count, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            label, r = fut.result()
            df.at[i, "raw"] = r["content"]
            df.at[i, "endorse_response"] = label
            for col in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "served_model"):
                df.at[i, col] = r[col]
            capped = guard.add(r)
            if count % checkpoint == 0:
                df.to_csv(out_path, index=False)
            if capped:
                for f in futures:
                    f.cancel()
                print(f"[endorse] --max-cost ${max_cost} reached; stopping (resumable)")
                break
    df.to_csv(out_path, index=False)
    print(f"[endorse] spend ${guard.spent:.3f}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    choices=("attempt1", "gateloop", "gated", "human", "csv"))
    ap.add_argument("--name", required=True, help="output stem, e.g. oeq_base_finaldata")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--tag", default=None, help="e.g. base_finaldata_temp1.0_seed0")
    ap.add_argument("--expected-shards", type=int, default=None)
    ap.add_argument("--csv", default=None, help="input csv for --source csv")
    ap.add_argument("--expect", type=int, default=None,
                    help="assert this many rows (1000 for attempt1/human)")
    ap.add_argument("--prompt-version", default="v5", choices=("v1", "v5"))
    ap.add_argument("--scores-root", default=DEFAULT_SCORES_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", type=int, default=50)
    ap.add_argument("--max-cost", type=float, default=5.0)
    args = ap.parse_args()

    if args.source in ("attempt1", "gateloop", "gated"):
        assert args.run_dir and args.tag, f"--source {args.source} needs --run-dir and --tag"
    if args.source == "attempt1":
        df = attempt1_frame(args.run_dir, args.tag, args.expected_shards)
    elif args.source == "gateloop":
        df = gateloop_frame(args.run_dir, args.tag)
    elif args.source == "gated":
        df = gated_frame(args.run_dir, args.tag, args.expected_shards)
    elif args.source == "human":
        df = human_frame()
    else:
        assert args.csv, "--source csv needs --csv"
        df = pd.read_csv(args.csv)
        if "id" not in df.columns:
            df.insert(0, "id", range(len(df)))
        df = df[["id", "sentence", "response"]]
    if args.expect is not None:
        assert len(df) == args.expect, f"expected {args.expect} rows, got {len(df)}"

    subdir = {"gateloop": f"endorse_gpt5mini_{args.prompt_version}_temp0_gateloop",
              "gated": f"endorse_gpt5mini_{args.prompt_version}_gated"}.get(
                  args.source, f"endorse_gpt5mini_{args.prompt_version}_temp0")
    inputs_dir = os.path.join(args.scores_root, "inputs")
    out_dir = os.path.join(args.scores_root, subdir)
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(inputs_dir, f"{args.name}.csv"), index=False)
    out_path = os.path.join(out_dir, f"scored_{args.name}.csv")
    scored = score(df, args.prompt_version, out_path, args.workers, args.checkpoint,
                   args.max_cost)

    labels = scored["raw"].map(first_endorse_label)
    rate, ci, n01 = action_endorsement_rate(labels.tolist())
    counts = {k: int((labels == k).sum()) for k in (0, 1, 2, 3)}
    print(f"[endorse] {args.name}: N={len(scored)} counts 0/1/2/3 = "
          f"{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]} invalid={int(labels.isna().sum())}")
    print(f"[endorse] action endorsement rate #1/(#0+#1) = {rate:.3f} +/- {ci:.3f} (n01={n01})")
    print(f"[endorse] -> {out_path}")


if __name__ == "__main__":
    main()
