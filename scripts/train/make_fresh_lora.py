"""Save a fresh, untrained LoRA adapter on a base model (e.g. Qwen3-32B) so
train_offline.py can train it from scratch on a pair file.

Geometry matches the stage-1 adapters (r=16, alpha=32, dropout 0.05, the seven
attention+MLP projections). LoRA B is zero at init, so the adapter starts
exactly at the base model.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plurpo.models import load_model

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the LoRA A init")
    args = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    assert not os.path.exists(os.path.join(args.out_dir, "adapter_config.json")), (
        f"adapter already exists at {args.out_dir}")
    model, tokenizer = load_model(args.model_id, device=args.device, cache_dir=args.cache_dir)
    torch.manual_seed(args.seed)
    peft_model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM", init_lora_weights=True, use_rslora=False, use_dora=False))
    os.makedirs(args.out_dir, exist_ok=True)
    peft_model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[fresh-lora] {args.model_id} -> {args.out_dir}")


if __name__ == "__main__":
    main()
