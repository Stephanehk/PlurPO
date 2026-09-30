"""Qwen3-8B + per-response-GEPA (App. gepa_1).

dspy.GEPA (auto="heavy", seed 0; ~8000 metric calls) searches Qwen3-8B's
instruction so each training prompt's response gets the same v5
action-endorsement label as the paired human response (score 1 on a match).
Train pool: data/splits/oeq_train (1000); Pareto/validation pool:
data/tuning/oeq_test (1000). The optimised program is then run once over
data/splits/oeq_test at the held-out decode settings and judged.

In the paper's run GEPA never accepted a revision, so the optimised program
is the seed instruction. (That run's research output dir was
dspy_gepa_oeq_origdata_full_fixedmetric; its held-out match rate was 0.467.)

Requires OPENROUTER_API_KEY and one GPU.
Usage: python scripts/baselines/gepa/run_per_response.py --out-dir runs/gepa_per_response
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
import pandas as pd

from gepa_lib import (BudgetTracker, RespondProgram, human_labelled, judge_endorse,
                      labelled_examples, make_oeq_metric, make_qwen_lm, make_reflection_lm,
                      oeq_frame, start_qwen_server)
from plurpo.llm_api import make_client

ARM = "oeq_origdata_full"


def evaluate_on_heldout(program, test_df, client, tracker, eval_n=None):
    """Per-prompt human-label match rate of the optimised program on the
    held-out set."""
    if eval_n is not None:
        test_df = test_df.iloc[:eval_n]
    rows = []
    for r in test_df.itertuples():
        pred = program(post_text=r.post_text)
        model_label = judge_endorse(client, tracker, r.post_text, pred.response, arm=f"{ARM}_eval")
        rows.append({"id": r.id, "human_label": r.human_label, "model_label": model_label,
                     "match": int(model_label == r.human_label)})
    df = pd.DataFrame(rows)
    return df, {"metric": "per_prompt_endorsement_match_rate",
                "value": float(df["match"].mean()), "n": len(df)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--auto", default="heavy", choices=("light", "medium", "heavy"))
    ap.add_argument("--eval-n", type=int, default=None, help="debug: cap held-out rows")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tracker = BudgetTracker(os.path.join(args.out_dir, "budget_state.json"))
    client = make_client()
    cache = lambda name: os.path.join(args.out_dir, f"human_labels_{name}.csv")
    train_df = human_labelled(oeq_frame("train", "final"), client, tracker,
                              cache("final_train"), arm="oeq_train_humanlabel")
    val_df = human_labelled(oeq_frame("test", "tuning"), client, tracker,
                            cache("tuning_test"), arm=f"{ARM}_valset_humanlabel")
    trainset, valset = labelled_examples(train_df), labelled_examples(val_df)
    print(f"[data] train_pool={len(trainset)} val_pool={len(valset)}")

    proc, base_url = start_qwen_server(args.device, os.path.join(args.out_dir, "vllm.log"))
    dspy.configure(lm=make_qwen_lm(base_url, mode="search"))
    optimizer = dspy.GEPA(metric=make_oeq_metric(client, tracker, arm=ARM),
                          reflection_lm=make_reflection_lm(tracker, f"{ARM}_reflection"),
                          track_stats=True, log_dir=os.path.join(args.out_dir, "gepa_log"),
                          seed=0, auto=args.auto)
    t0 = time.time()
    optimized = optimizer.compile(RespondProgram(), trainset=trainset, valset=valset)
    print(f"[gepa] compile finished in {time.time() - t0:.0f}s")
    optimized.save(os.path.join(args.out_dir, "optimized_program.json"))

    dspy.configure(lm=make_qwen_lm(base_url, mode="eval"))
    test_df = human_labelled(oeq_frame("test", "final"), client, tracker,
                             cache("final_test"), arm=f"{ARM}_test_humanlabel")
    results, summary = evaluate_on_heldout(optimized, test_df, client, tracker, args.eval_n)
    results.to_csv(os.path.join(args.out_dir, "heldout_results.csv"), index=False)
    summary.update(env=ARM, test_stem="oeq", max_metric_calls=optimizer.max_metric_calls)
    with open(os.path.join(args.out_dir, "heldout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] {summary}; spend ${tracker.snapshot()['spent_usd']:.4f}")
    proc.terminate()


if __name__ == "__main__":
    main()
