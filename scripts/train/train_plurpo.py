"""PlurPO stage 1: iterative stakeholder-veto preference optimisation.

Two GPUs: the trainable policy M1 (HF model + LoRA) on --device, and one vLLM
engine on --sim-device that serves BOTH the frozen stakeholder simulator M2
(base weights, greedy) and the policy sampler (base + the current LoRA,
hot-loaded before every epoch). Nothing here calls an external API.

Outputs in --output-dir: adapter_iter_<k>/ per epoch, the final adapter,
responses_iter_<k>.jsonl (every candidate, its source, veto and exclusion
status), iter_stats.json, train_ids.json / test_ids.json.

Usage:
  python scripts/train/train_plurpo.py --config qwen3-8b --env OEQ \
      --output-dir runs/plurpo_qwen3-8b_oeq_stage1

  # PlurPO-no-stakeholder-preferences ablation (App. "Ablating PlurPO"):
  python scripts/train/train_plurpo.py --config qwen3-8b --env OEQ \
      --output-dir runs/pairpref_oeq_stage1 --pair-source preference --grad-accum 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.candidates import CandidateSampler
from plurpo.config import load_config, per_env
from plurpo.data import ENVS, load_split, train_prompts
from plurpo.online import OnlineTrainer
from plurpo.rpo import dpo_config_kwargs
from plurpo.runner import build_policy, engine_ready, start_engine, write_split_ids
from plurpo.stakeholders import StakeholderSimulator, make_sim_call
from plurpo.vllm_server import make_vllm_call_llm, make_vllm_policy_sampler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<name>.json, e.g. qwen3-8b")
    ap.add_argument("--env", required=True, choices=ENVS)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pair-source", default="veto", choices=("veto", "preference"),
                    help="'preference' = PlurPO-no-stakeholder-preferences ablation")
    ap.add_argument("--preference-max-pairs", type=int, default=10)
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override the config's grad_accum. The paper's ablation "
                         "run used 8 (effective batch 8, vs 16 for PlurPO itself)")
    ap.add_argument("--device", default="cuda:0", help="GPU for the trainable policy")
    ap.add_argument("--sim-device", default="cuda:1", help="GPU for the vLLM engine")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N prompts only")
    ap.add_argument("--num-iterations", type=int, default=None, help="debug override")
    ap.add_argument("--rpo-alpha", type=float, default=None,
                    help="override the config's alpha (App. alpha sweep; <= 0 = plain DPO)")
    ap.add_argument("--split-set", default="final", choices=("final", "tuning"),
                    help="'tuning' = the held-out OEQ/PAS split PlurPO was tuned on")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    if args.grad_accum is not None:
        s1["grad_accum"] = args.grad_accum
    if args.num_iterations is not None:
        s1["num_iterations"] = args.num_iterations
    if args.rpo_alpha is not None:
        s1["rpo_alpha"] = args.rpo_alpha if args.rpo_alpha > 0 else None
    assert args.sim_device != args.device, "the vLLM engine needs its own GPU"
    os.makedirs(args.output_dir, exist_ok=True)
    write_split_ids(args.env, args.output_dir, args.split_set)
    prompts = train_prompts(args.env, load_split(args.env, "train", args.split_set))
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"[data] {args.env}: {len(prompts)} training prompts")

    proc, url, log = start_engine(cfg["model_id"], args.sim_device,
                                  s1["vllm_gpu_memory_utilization"],
                                  cfg["vllm_max_model_len"], s1["lora_r"], args.output_dir)
    model, tokenizer, start = build_policy(
        cfg["model_id"], cfg["lora_target_modules"], s1["lora_r"], s1["lora_alpha"],
        s1["lora_dropout"], s1["num_iterations"], args.device, args.cache_dir,
        args.output_dir)
    template_kwargs, sync_policy = engine_ready(proc, url, log, cfg["model_id"],
                                                args.cache_dir)

    judge = StakeholderSimulator(
        make_sim_call(make_vllm_call_llm(url, cfg["model_id"],
                                         chat_template_kwargs=template_kwargs)),
        veto_prompt=s1["veto_prompt"], user_stakeholder=s1["user_stakeholder"],
        stakeholder_max_tokens=s1["stakeholder_max_tokens"],
        veto_max_tokens=s1["veto_max_tokens"])
    sampler = CandidateSampler(
        make_vllm_policy_sampler(url, cfg["model_id"], chat_template_kwargs=template_kwargs),
        n_samples=s1["n_samples"], n_model=s1["n_model"], n_steer=s1["n_steer"],
        steer=s1["steer"], perspshift_assess=per_env(s1["perspshift_assess"], args.env),
        temperature=s1["temperature"], top_p=s1["top_p"],
        max_tokens=s1["sample_max_tokens"], rewrite_max_tokens=s1["rewrite_max_tokens"])
    dpo_kwargs = dpo_config_kwargs(
        s1["beta"], s1["rpo_alpha"], s1["learning_rate"], 1, s1["grad_accum"],
        s1["max_length"], s1["precompute_ref_batch_size"], s1["dpo_seed"],
        use_bf16=True)
    trainer = OnlineTrainer(model, tokenizer, judge, sampler, sync_policy, dpo_kwargs,
                            pair_source=args.pair_source,
                            preference_max_pairs=args.preference_max_pairs,
                            chunk_size=s1["chunk_size"], concurrency=s1["concurrency"])
    trainer.train(prompts, args.output_dir, s1["num_iterations"], start_iteration=start)
    print(f"[done] {judge.n_excluded} candidates excluded (unreadable stance/vote); "
          f"adapter -> {args.output_dir}")
    proc.terminate()


if __name__ == "__main__":
    main()
