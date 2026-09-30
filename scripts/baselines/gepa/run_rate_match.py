"""Qwen3-8B + endorsement-rate-GEPA (App. gepa_2).

gepa.optimize() with a custom adapter whose batch score is
1 - |model_rate - TARGET_RATE|, where model_rate is the batch's action
endorsement rate #1/(#0+#1) of Qwen3-8B's responses (a batch with no 0/1
label scores 0). Per-item scoring cannot express an interior target rate, so
RateMatchAdapter overrides only DspyAdapter.evaluate's numeric aggregation;
everything else is gepa's stock machinery.

Train pool = validation pool = data/splits/oeq_train (1000). TARGET_RATE =
0.5127 is the human responses' rate on data/splits/oeq_test (394 resolved of
1000). Stops after --patience proposals without a full-valset improvement
(the paper's run halted after 10), --max-proposals, or --budget-usd.

The research run was launched by hand as:
  run_gepa_oeq_ratematch.py --reflection-minibatch-size 16 --patience 10
      --max-proposals 40 --budget-usd 20
which are this script's defaults. Requires OPENROUTER_API_KEY and one GPU.
Usage: python scripts/baselines/gepa/run_rate_match.py --out-dir runs/gepa_rate_match
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dspy
import pandas as pd
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa import optimize
from gepa.utils import MaxCandidateProposalsStopper, NoImprovementStopper

from gepa_lib import (BudgetTracker, RespondProgram, judge_endorse, make_oeq_resolved_metric,
                      make_qwen_lm, make_reflection_lm, oeq_frame, post_examples,
                      start_qwen_server)
from plurpo.llm_api import make_client

TARGET_RATE = 0.5127
ARM = "oeq_ratematch"


class RateMatchAdapter(DspyAdapter):
    """Replaces per-item scores with one uniform batch score
    1 - |model_rate - target_rate| over the batch's resolved {0,1} labels;
    0.0 if none resolve."""

    def __init__(self, *args, target_rate, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_rate = target_rate

    def evaluate(self, batch, candidate, capture_traces=False):
        eval_batch = super().evaluate(batch, candidate, capture_traces=capture_traces)
        resolved = [s for s in eval_batch.scores if s in (0.0, 1.0)]
        if resolved:
            model_rate = sum(1.0 for s in resolved if s == 1.0) / len(resolved)
            uniform_score = 1.0 - abs(model_rate - self.target_rate)
        else:
            uniform_score = 0.0
        eval_batch.scores = [uniform_score] * len(eval_batch.scores)
        return eval_batch


class RunBudgetStopper:
    """Stop once THIS run's marginal spend reaches `budget_usd`."""

    def __init__(self, tracker, budget_usd):
        self.tracker = tracker
        self.budget_usd = budget_usd
        self.start_spent = tracker.snapshot()["spent_usd"]

    def __call__(self, gepa_state):
        return self.tracker.snapshot()["spent_usd"] - self.start_spent >= self.budget_usd


def evaluate_on_heldout(program, test_df, client, tracker):
    rows = []
    for r in test_df.itertuples():
        pred = program(post_text=r.post_text)
        rows.append({"id": r.id, "model_label": judge_endorse(
            client, tracker, r.post_text, pred.response, arm=f"{ARM}_eval")})
    df = pd.DataFrame(rows)
    resolved = df[df["model_label"].isin([0, 1])]
    model_rate = float((resolved["model_label"] == 1).mean()) if len(resolved) else None
    return df, {"metric": "action_endorsement_rate_gap", "model_rate": model_rate,
                "target_rate": TARGET_RATE,
                "gap": abs(model_rate - TARGET_RATE) if model_rate is not None else None,
                "n_resolved": int(len(resolved)), "n": len(df)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reflection-minibatch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-proposals", type=int, default=40)
    ap.add_argument("--budget-usd", type=float, default=20.0)
    ap.add_argument("--eval-n", type=int, default=None, help="debug: cap held-out rows")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tracker = BudgetTracker(os.path.join(args.out_dir, "budget_state.json"))
    client = make_client()
    trainset = post_examples(oeq_frame("train", "final"))
    student = RespondProgram()

    proc, base_url = start_qwen_server(args.device, os.path.join(args.out_dir, "vllm.log"))
    dspy.configure(lm=make_qwen_lm(base_url, mode="search"))
    reflection_lm = make_reflection_lm(tracker, f"{ARM}_reflection")
    metric_fn = make_oeq_resolved_metric(client, tracker, arm=ARM)
    predictors = list(student.named_predictors())
    assert len(predictors) == 1, predictors

    def feedback_fn_creator(pred_name, predictor):
        def feedback_fn(predictor_output, predictor_inputs, module_inputs, module_outputs,
                        captured_trace):
            trace_for_pred = [(predictor, predictor_inputs, predictor_output)]
            return metric_fn(module_inputs, module_outputs, captured_trace, pred_name,
                             trace_for_pred)
        return feedback_fn

    adapter = RateMatchAdapter(
        student_module=student, metric_fn=metric_fn,
        feedback_map={name: feedback_fn_creator(name, p) for name, p in predictors},
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size, target_rate=TARGET_RATE)
    stoppers = [NoImprovementStopper(max_iterations_without_improvement=args.patience),
                MaxCandidateProposalsStopper(max_proposals=args.max_proposals),
                RunBudgetStopper(tracker, args.budget_usd)]
    t0 = time.time()
    result = optimize(
        seed_candidate={name: p.signature.instructions for name, p in predictors},
        trainset=trainset, valset=trainset, adapter=adapter,
        reflection_lm=(lambda x: adapter.stripped_lm_call(x)[0]),
        candidate_selection_strategy="pareto", skip_perfect_score=True,
        reflection_minibatch_size=args.reflection_minibatch_size,
        module_selector="round_robin", use_merge=True, max_merge_invocations=5,
        stop_callbacks=stoppers, run_dir=os.path.join(args.out_dir, "gepa_log"),
        display_progress_bar=True, seed=0)
    print(f"[gepa] finished in {time.time() - t0:.0f}s, best full-valset score "
          f"{result.val_aggregate_scores[result.best_idx]:.4f}")
    optimized = adapter.build_program(result.best_candidate)
    optimized.save(os.path.join(args.out_dir, "optimized_program.json"))

    dspy.configure(lm=make_qwen_lm(base_url, mode="eval"))
    test_df = oeq_frame("test", "final")
    if args.eval_n is not None:
        test_df = test_df.iloc[:args.eval_n]
    results, summary = evaluate_on_heldout(optimized, test_df, client, tracker)
    results.to_csv(os.path.join(args.out_dir, "heldout_results.csv"), index=False)
    with open(os.path.join(args.out_dir, "heldout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] {summary}")
    proc.terminate()


if __name__ == "__main__":
    main()
