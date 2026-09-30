"""Local HuggingFace model loading and chat formatting.

Every model in the paper (Qwen3-8B, Qwen3-32B, Phi-4, Llama-3.1-8B-Instruct,
Granite-4.1-8B) is an `AutoModelForCausalLM` checkpoint with a chat template.
All prompts are rendered through the model's own chat template with no system
turn and `enable_thinking=False` (Qwen3's switch to disable <think> blocks;
templates that do not reference the variable ignore it).

Assumption (asserted at load time): the chat template survives
render-to-string-then-tokenize. Sampling and DPO training both depend on it,
because TRL's DPO dataset API is string-based.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"


def assert_chat_template_round_trips(tokenizer, model_id):
    """Fail loudly unless tokenize(render(x)) == apply_chat_template(tokenize=True)."""
    for prompt in ("hello world", "My friend ghosted me. AITA for being upset?"):
        messages = [{"role": "user", "content": prompt}]
        direct = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            enable_thinking=False)
        direct = list(direct["input_ids"]) if hasattr(direct, "keys") else list(direct)
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        round_trip = list(tokenizer(text, add_special_tokens=False).input_ids)
        assert direct == round_trip, (
            f"{model_id}: chat template does not survive render-then-retokenize "
            f"({len(direct)} vs {len(round_trip)} tokens)")


def load_model(model_id=DEFAULT_MODEL_ID, device="cuda:0", cache_dir=None,
               device_map=None):
    """Load (model, tokenizer) in bf16, in eval mode.

    With `device_map=None` the model is placed on the single `device`; with a
    `device_map` (e.g. "auto", used to train Qwen3-32B across two GPUs)
    accelerate places the shards and `.to(device)` is skipped. Tokenizers
    without a pad token reuse EOS.
    """
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    assert_chat_template_round_trips(tokenizer, model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
        device_map=device_map)
    if device_map is None:
        model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def render_chat(tokenizer, prompt, system=None):
    """Render one user turn (or a prebuilt list of {"role","content"} turns,
    e.g. k-shot exemplars followed by the target turn) to the templated
    string, with the generation prompt appended."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(prompt)
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False)


def build_chat_input(tokenizer, prompt, system, device):
    """Templated input_ids (1xN LongTensor on `device`) for `prompt`.
    `add_special_tokens=False` because the template already inserts BOS and
    role markers."""
    text = render_chat(tokenizer, prompt, system)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc.input_ids.to(device)


def retemplate_prompt(tokenizer, prompt):
    """The DPO `prompt` column for a bare user prompt: the same chat-templated
    prefix the candidates were sampled under, so DPO scores
    "<chat-template> -> response" rather than "raw text -> response"."""
    return render_chat(tokenizer, prompt)
