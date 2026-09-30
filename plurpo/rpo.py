"""Iterative RPO / DPO training of a LoRA adapter on preference pairs (Sec. 3.2).

Loss (Pang et al., 2024): L = L_DPO(beta) + alpha * NLL(chosen), with the NLL
term mean-per-token over the chosen completion. Implemented with TRL's
multi-loss API: loss_type=["sigmoid", "sft"], loss_weights=[1.0, alpha].
alpha=None gives plain DPO. The reference policy is the adapter-DISABLED base
model (TRL's behaviour for a PeftModel with ref_model=None), for every epoch
and for the role-confusion round.

Batching: one pair per micro-batch, always (the per-device train batch is
asserted). The effective batch is `grad_accum`. The paper's online stage-1
runs used an effective batch of 16 (see configs/); the offline rounds used 8.
"""

import os

import torch

from plurpo.models import retemplate_prompt

SENTINEL_NAME = ".complete"


class NonFiniteAdapterError(RuntimeError):
    """A LoRA adapter contains NaN/Inf weights; never save or serve it."""


def assert_adapter_finite(model, where):
    """Raise if any trainable weight is NaN/Inf. Called before every adapter
    save and every policy sync to vLLM, so a diverged run leaves no loadable
    artifact behind."""
    bad = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            n_bad = int(torch.isnan(p.data).sum()) + int(torch.isinf(p.data).sum())
            if n_bad:
                bad.append((name, n_bad, p.numel()))
    if bad:
        detail = "\n".join(f"    {n}: {k} non-finite of {m}" for n, k, m in bad[:10])
        raise NonFiniteAdapterError(
            f"NON-FINITE LoRA WEIGHTS at {where}: {len(bad)} tensors affected; "
            f"refusing to save/serve.\n{detail}")


def write_complete_sentinel(adapter_dir):
    """Mark `adapter_dir` as fully saved (call right after save_pretrained)."""
    open(os.path.join(adapter_dir, SENTINEL_NAME), "w").close()


def adapter_complete(adapter_dir):
    """True iff `adapter_dir` carries the `.complete` sentinel."""
    return os.path.exists(os.path.join(adapter_dir, SENTINEL_NAME))


def save_adapter(model, out_dir, where):
    """Finite-check, save the LoRA adapter, then write the sentinel."""
    assert_adapter_finite(model, where)
    model.save_pretrained(out_dir)
    write_complete_sentinel(out_dir)


def dpo_config_kwargs(beta, rpo_alpha, learning_rate, num_train_epochs, grad_accum,
                      max_length, precompute_ref_batch_size, seed, use_bf16):
    """DPOConfig keyword arguments shared by every PlurPO training update.

    Reference log-probs are precomputed in a separate no-grad pass (the
    in-step reference forward materialises a full-vocabulary logits tensor and
    OOMs on long OEQ sequences). No checkpoints are written mid-epoch.
    """
    kwargs = dict(
        beta=float(beta),
        learning_rate=float(learning_rate),
        num_train_epochs=float(num_train_epochs),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(grad_accum),
        max_length=int(max_length),
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=int(precompute_ref_batch_size),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        remove_unused_columns=False,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        seed=int(seed),
    )
    if rpo_alpha is not None:
        assert rpo_alpha > 0, f"rpo_alpha must be > 0 when set (got {rpo_alpha})"
        kwargs["loss_type"] = ["sigmoid", "sft"]
        kwargs["loss_weights"] = [1.0, float(rpo_alpha)]
    return kwargs


def train_on_pairs(model, tokenizer, pairs, output_dir, config_kwargs):
    """Run one DPO/RPO update of the (already LoRA-wrapped, trainable) `model`
    on `pairs` ({"prompt","chosen","rejected"} with BARE prompts).

    The prompt column is re-rendered through the chat template the candidates
    were sampled under. `_n_gpu` is forced to 1 before the trainer is built so
    HF never wraps the policy in DataParallel (the vLLM server owns the other
    GPU) and never multiplies the per-device batch by the GPU count.
    Returns the DPOTrainer.
    """
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    assert pairs, "no pairs to train on"
    dataset = Dataset.from_list([
        {"prompt": retemplate_prompt(tokenizer, p["prompt"]),
         "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs])
    args = DPOConfig(output_dir=output_dir, **config_kwargs)
    args._n_gpu = 1
    model.train()
    trainer = DPOTrainer(model=model, ref_model=None, args=args,
                         train_dataset=dataset, processing_class=tokenizer,
                         peft_config=None)
    assert trainer._train_batch_size == 1, (
        f"per-device train batch is {trainer._train_batch_size}, expected 1; "
        f"transformers changed how it is derived")
    print(f"[rpo] {len(pairs)} pairs, effective batch "
          f"{trainer._train_batch_size * args.gradient_accumulation_steps}")
    trainer.train()
    return trainer
