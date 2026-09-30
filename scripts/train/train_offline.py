"""One offline RPO/DPO epoch on a precomputed pair file.

Used for:
  - PlurPO stage 2 (role confusion): continue the stage-1 adapter on
    <out-dir>/pairs/pairs.jsonl;
  - Qwen3-32B: train a fresh LoRA (made by make_fresh_lora.py) on the
    Qwen3-8B stage-1 pairs in data/plurpo_qwen3-8b_pairs/;
  - the offline baselines (Inferred-Prefs-DPO).

The reference policy is the adapter-disabled base model, so the KL term pulls
toward the base, exactly as in stage 1. The input adapter is never written
to; the result is saved to --out-dir with a .complete sentinel. Only the
prompt/chosen/rejected keys of each pair are used.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.io import read_jsonl
from plurpo.models import DEFAULT_MODEL_ID, load_model
from plurpo.rpo import dpo_config_kwargs, save_adapter, train_on_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="LoRA adapter to continue (read-only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs-file", default=None, help="default <out-dir>/pairs/pairs.jsonl")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default=None, help='"auto" to shard across GPUs (Qwen3-32B)')
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=0.5, help="<= 0 for plain DPO")
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--precompute-ref-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from peft import PeftModel

    assert os.path.exists(os.path.join(args.adapter, "adapter_config.json")), args.adapter
    assert os.path.normpath(args.adapter) != os.path.normpath(args.out_dir), (
        "--out-dir must differ from --adapter")
    pairs_file = args.pairs_file or os.path.join(args.out_dir, "pairs", "pairs.jsonl")
    pairs = [{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in read_jsonl(pairs_file)]
    print(f"[offline] {len(pairs)} pairs from {pairs_file}")

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model_id, device=args.device,
                                  cache_dir=args.cache_dir, device_map=args.device_map)
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0

    kwargs = dpo_config_kwargs(
        args.beta, args.rpo_alpha if args.rpo_alpha > 0 else None, args.learning_rate,
        args.num_train_epochs, args.grad_accum, args.max_length,
        args.precompute_ref_batch_size, args.seed, use_bf16=True)
    trainer = train_on_pairs(model, tokenizer, pairs, os.path.join(args.out_dir, "dpo"), kwargs)
    os.makedirs(args.out_dir, exist_ok=True)
    save_adapter(model, args.out_dir, "offline adapter")
    tokenizer.save_pretrained(args.out_dir)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if losses:
        print(f"[offline] first loss={losses[0]:.4f} last loss={losses[-1]:.4f}")
    print(f"[offline] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
