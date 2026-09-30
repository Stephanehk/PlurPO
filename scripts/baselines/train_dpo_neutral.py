"""DPO-Neutral baseline (Sec. 4.3, App. "Neutral-DPO Implementation Details").

3 on-policy epochs. Per training prompt, 10 candidates: 5 from the bare post
and 5 from the post + "Be neutral in your response. Don't endorse or
critique."; the gpt-5-mini v5 endorsement judge labels each, and every
neutral-labelled candidate is preferred to every other candidate. Plain DPO
(no NLL term), otherwise the PlurPO hyperparameters.

Batch: grad_accum 16 reproduces the paper's effective batch of 16 (these runs
used per-device batch 2 x grad_accum 8 on 2-GPU nodes, before the batch-size
fix; inferred from the run dates and setup, not from a training log).

Two GPUs: policy on --device, the vLLM policy sampler on --sim-device.
Requires OPENROUTER_API_KEY. OEQ and PAS only (AITA verdicts cannot be neutral).

Usage:
  python scripts/baselines/train_dpo_neutral.py --env OEQ --output-dir runs/dpo_neutral_oeq
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.candidates import CandidateSampler
from plurpo.data import load_split, train_prompts
from plurpo.llm_api import make_client
from plurpo.neutrality import NeutralityJudge
from plurpo.online import OnlineTrainer
from plurpo.rpo import dpo_config_kwargs
from plurpo.runner import build_policy, engine_ready, start_engine, write_split_ids
from plurpo.vllm_server import make_vllm_policy_sampler

MODEL_ID = "Qwen/Qwen3-8B"
NUM_ITERATIONS = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=("OEQ", "PAS"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sim-device", default="cuda:1")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    write_split_ids(args.env, args.output_dir)
    prompts = train_prompts(args.env, load_split(args.env, "train"))

    proc, url, log = start_engine(MODEL_ID, args.sim_device, 0.90, None, 16, args.output_dir)
    model, tokenizer, start = build_policy(MODEL_ID, "all-linear", 16, 32, 0.05,
                                           NUM_ITERATIONS, args.device, args.cache_dir,
                                           args.output_dir)
    template_kwargs, sync_policy = engine_ready(proc, url, log, MODEL_ID, args.cache_dir)
    sampler = CandidateSampler(
        make_vllm_policy_sampler(url, MODEL_ID, chat_template_kwargs=template_kwargs),
        n_samples=10, n_model=5, n_steer=5, steer="neutral",
        temperature=1.0, top_p=0.95, max_tokens=1024)
    judge = NeutralityJudge(make_client(), concurrency=args.judge_concurrency)
    dpo_kwargs = dpo_config_kwargs(beta=0.1, rpo_alpha=None, learning_rate=5e-5,
                                   num_train_epochs=1, grad_accum=args.grad_accum,
                                   max_length=3072, precompute_ref_batch_size=2,
                                   seed=42, use_bf16=True)
    trainer = OnlineTrainer(model, tokenizer, judge, sampler, sync_policy, dpo_kwargs)
    trainer.train(prompts, args.output_dir, NUM_ITERATIONS, start_iteration=start)
    print(f"[done] {judge.n_invalid} unparsable judge labels (treated as non-neutral)")
    proc.terminate()


if __name__ == "__main__":
    main()
