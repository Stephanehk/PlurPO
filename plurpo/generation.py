"""Held-out generation helpers shared by the evaluation scripts.

Decoding contract (matches every generation behind the paper's numbers):
  - the bare post (optionally wrapped by a prompting baseline's steer, or
    replaced by its perspective-shift rewrite) is the single user turn,
    rendered with the model's chat template, no system turn,
    enable_thinking=False (plurpo.models.build_chat_input);
  - ONE sample per prompt at temperature 1.0, top_p 0.95;
  - `torch.manual_seed(seed)` ONCE per generation job (per shard), so attempt-1
    draws depend on the shard layout (OEQ/PAS/FLIP 2 shards, AITA 1 in the
    paper runs);
  - rejection-sampling retries k = 2..MAX_ATTEMPTS are instead seeded per
    (prompt, attempt) by `attempt_seed`, so a retry is a pure function of what
    is being sampled and which retry it is.

Tag convention (every output filename): `<adapter_tag>_finaldata_temp1.0_seed0`,
where adapter_tag is "base" / "final" (the run dir's own final adapter) / the
basename of another adapter dir (e.g. "adapter_iter_2"), followed by the steer
suffix ("-becritical", "-explicit", "-sharma", "-dontbesyco", "-stakeholder",
"-perspshiftassess", "-<k>shothuman").
"""

import hashlib
import os
import sys

import torch

from plurpo.models import build_chat_input
from plurpo.prompts.baselines import STEER_TAGS

MAX_ATTEMPTS = 5


def steer_suffix_tag(steer):
    """Tag suffix for a plain steer ("" for none): STEER_TAGS minus "base"."""
    tag = STEER_TAGS[steer]
    assert tag.startswith("base"), f"unexpected steer tag {tag!r}"
    return tag[len("base"):]


def adapter_tag_for(base, adapter_dir, run_dir):
    """"base" for the untrained model, "final" when evaluating the run dir's
    own adapter, else the adapter dir's basename."""
    if base:
        return "base"
    if os.path.normpath(adapter_dir) == os.path.normpath(run_dir):
        return "final"
    return os.path.basename(os.path.normpath(adapter_dir))


def build_tag(adapter_tag, temperature, seed, greedy=False):
    """Full output tag for the frozen final split."""
    decode_tag = "greedy" if greedy else f"temp{temperature}"
    return f"{adapter_tag}_finaldata_{decode_tag}_seed{seed}"


def attempt_seed(prompt, attempt, base_seed=0):
    """Deterministic torch seed for one (prompt, attempt) draw: sha256 of
    "<base_seed>|<attempt>|<prompt>", truncated to 63 bits (torch.manual_seed
    rejects values outside a signed 64-bit int)."""
    h = hashlib.sha256(f"{base_seed}|{attempt}|{prompt}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2 ** 63)


def shard_slice(n, num_shards, shard_id):
    """(start, end) of block `shard_id` when [0, n) is split into `num_shards`
    contiguous near-equal blocks (the first n % num_shards get one extra)."""
    assert num_shards >= 1 and 0 <= shard_id < num_shards, (
        f"bad shard ({shard_id} of {num_shards})")
    base, rem = divmod(n, num_shards)
    sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]
    start = sum(sizes[:shard_id])
    return start, start + sizes[shard_id]


def attach_adapter(base_model, adapter_dir):
    """Wrap the base model with the LoRA adapter in `adapter_dir` (eval mode)."""
    assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json")), (
        f"no adapter_config.json in {adapter_dir} -- not a PEFT adapter dir")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


def generate_one(model, tokenizer, prompt, max_new_tokens, temperature, top_p, greedy):
    """One response for `prompt` (a string or a list of chat turns), decoded
    with skip_special_tokens and stripped.

    Guard: multi-GPU (`device_map`) generation can rarely emit an out-of-range
    token id, which crashes decode. Such a draw is regenerated ONCE; a second
    out-of-range draw is systematic and raises.
    """
    device = next(model.parameters()).device
    input_ids = build_chat_input(tokenizer, prompt, system=None, device=device)
    in_len = input_ids.shape[1]
    gen_kwargs = dict(max_new_tokens=int(max_new_tokens), num_return_sequences=1,
                      do_sample=not greedy, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)
    if not greedy:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    vocab_n = len(tokenizer)
    for attempt in range(2):
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)
        new_ids = output_ids[0, in_len:]
        bad = (new_ids < 0) | (new_ids >= vocab_n)
        if not bool(bad.any()):
            return tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"[generate_one] WARNING: {int(bad.sum())} out-of-range token id(s) "
              f"on attempt {attempt + 1}/2", file=sys.stderr, flush=True)
    raise AssertionError(
        f"generate_one: token ids out of range [0,{vocab_n}) on both attempts -- "
        f"investigate device placement or the checkpoint")


def resample_until_pass(gen_fn, judge, prompt_key, context, base_seed,
                        max_attempts=MAX_ATTEMPTS, log=print):
    """Rejection sampling for one prompt whose attempt-1 draw the gate rejected.

    For attempt = 2..max_attempts: reseed with attempt_seed(prompt_key, attempt,
    base_seed), draw with `gen_fn()`, stop at the first draw `judge(context,
    draw)` labels 1. The gate always sees the bare ORIGINAL post as `context`.

    Returns (response, winning_attempt, n_tried): the passing draw and its
    attempt index, or -- when every draw fails -- the LAST draw (kept for
    inspection, never scored), None, and max_attempts.

    (The research driver made one extra, discarded draw seeded exactly like
    attempt 2 before entering this loop; it had no effect on any kept draw, so
    it is omitted here.)
    """
    assert max_attempts >= 2, "max_attempts must leave room for at least one resample"
    last = None
    for attempt in range(2, max_attempts + 1):
        torch.manual_seed(attempt_seed(prompt_key, attempt, base_seed))
        resp = gen_fn()
        last = resp
        if judge(context, resp) == 1:
            log(f"    passed at attempt {attempt}")
            return resp, attempt, attempt
    log(f"    EXHAUSTED after {max_attempts} attempts -- counted as a gate failure")
    return last, None, max_attempts
