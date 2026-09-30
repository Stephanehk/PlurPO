"""Setup shared by the on-policy training entry points (PlurPO stage 1 and
DPO-Neutral): the trainable LoRA policy, the vLLM engine that samples from it
(and, for PlurPO, also serves the frozen simulator), and split bookkeeping.
"""

import json
import os

from plurpo.candidates import POLICY_ADAPTER
from plurpo.config import lora_targets
from plurpo.data import load_ids
from plurpo.models import load_model
from plurpo.online import detect_resume_point
from plurpo.vllm_server import (chat_template_kwargs_for, launch_vllm_server,
                                load_lora_adapter, unload_lora_adapter,
                                wait_for_vllm_server)


def build_policy(model_id, lora_target_modules, lora_r, lora_alpha, lora_dropout,
                 num_iterations, device, cache_dir, output_dir):
    """Load M1 on `device` and wrap it with a fresh LoRA, or resume the last
    finished epoch's adapter. Returns (model, tokenizer, start_epoch)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    model, tokenizer = load_model(model_id, device=device, cache_dir=cache_dir)
    model.enable_input_require_grads()
    start = detect_resume_point(output_dir, num_iterations)
    if start > 0:
        adapter = os.path.join(output_dir, f"adapter_iter_{start}")
        print(f"[resume] continuing from {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_targets(lora_target_modules),
            bias="none", task_type="CAUSAL_LM"))
    return model, tokenizer, start


def start_engine(model_id, device, gpu_memory_utilization, max_model_len, lora_r,
                 output_dir):
    """Launch (without waiting) the vLLM engine with LoRA hot-loading enabled.
    Returns (proc, base_url, log_path)."""
    log = os.path.join(output_dir, "vllm_server.log")
    proc, url = launch_vllm_server(model_id, device,
                                   gpu_memory_utilization=gpu_memory_utilization,
                                   max_model_len=max_model_len, log_path=log,
                                   enable_lora=True, max_lora_rank=lora_r)
    return proc, url, log


def engine_ready(proc, url, log, model_id, cache_dir):
    """Wait for the engine; return (chat_template_kwargs, sync_policy) where
    sync_policy(adapter_dir) hot-swaps the "policy" LoRA."""
    wait_for_vllm_server(proc, url, log_path=log)

    def sync_policy(adapter_dir):
        unload_lora_adapter(url, POLICY_ADAPTER)
        load_lora_adapter(url, POLICY_ADAPTER, os.path.abspath(adapter_dir))

    return chat_template_kwargs_for(model_id, cache_dir), sync_policy


def write_split_ids(env, output_dir, split_set="final"):
    """Record the split a run used, next to its adapters."""
    for split in ("train", "test"):
        with open(os.path.join(output_dir, f"{split}_ids.json"), "w") as f:
            json.dump(load_ids(env, split, split_set), f, indent=2)
