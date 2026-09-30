"""vLLM server helpers: the frozen stakeholder simulator (M2) and the policy
sampler (M1 = base + the current LoRA) share one engine.

The sim model (the frozen judge driving identify / filter / concerns /
veto in `plurpo/stakeholders.py`) never changes during a run, which
makes it a perfect fit for a vLLM server: continuous batching lets many
short greedy veto calls run concurrently instead of one batch-1 HF
`generate` at a time.

Public surface:
  - `launch_vllm_server(...)`: start a vLLM OpenAI-compatible server as
    a subprocess pinned to one GPU. Returns (proc, base_url)
    immediately, WITHOUT waiting for the server to come up, so the
    caller can overlap server startup with its own model loading.
  - `wait_for_vllm_server(proc, base_url, ...)`: block until the
    server's /health endpoint answers (or fail loud if the subprocess
    dies / the timeout expires).
  - `make_vllm_call_llm(base_url, model_id)`: returns a callable with
    the harness `call_llm` signature
        (client, model_name, prompt, max_tokens, system=None)
            -> (text, input_tokens, output_tokens)
    routed to the server. Greedy (temperature=0) to match
    greedy HF decoding, and passes
    `enable_thinking=False` through `chat_template_kwargs` so Qwen3
    behaves identically to the HF `build_chat_input` path. The returned
    callable is thread-safe — fire it from many threads to exploit the
    server's continuous batching.

Assumptions:
  - `vllm` and `openai` are installed in the running interpreter's env
    (the server is launched via `sys.executable -m
    vllm.entrypoints.openai.api_server`).
  - `curl` is on PATH (used for /health polling so no exception
    handling is needed around connection-refused during startup).
  - The target GPU is otherwise free: vLLM preallocates
    `gpu_memory_utilization` of it.
"""

import atexit
import os
import socket
import subprocess
import sys
import time


def _free_port():
    """Bind an ephemeral port on localhost, release it, and return its
    number. Small race window between release and the server binding it,
    but in practice safe — and it lets two jobs share a node without a
    hardcoded-port collision.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _visible_device_for(device):
    """Map a torch device string ("cuda:1") to the value the child
    process's CUDA_VISIBLE_DEVICES must be set to so the child sees that
    one GPU as its device 0.

    Honors any masking already applied to THIS process (e.g. SLURM sets
    CUDA_VISIBLE_DEVICES per job): torch device indices are positions
    within the parent's visible list, not physical ids, so "cuda:1" must
    resolve to the parent list's second entry.
    """
    idx = int(device.split(":")[1]) if ":" in device else 0
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent is None or parent.strip() == "":
        return str(idx)
    visible = [v.strip() for v in parent.split(",")]
    assert idx < len(visible), (
        f"device {device!r} is out of range of CUDA_VISIBLE_DEVICES={parent!r}"
    )
    return visible[idx]


def launch_vllm_server(model_id,
                       device,
                       gpu_memory_utilization=0.90,
                       max_model_len=None,
                       log_path=None,
                       enable_lora=False,
                       max_lora_rank=16,
                       dtype="bfloat16"):
    """Start a vLLM OpenAI-compatible server for `model_id` pinned to
    `device` ("cuda:N"). Returns (proc, base_url) IMMEDIATELY — call
    `wait_for_vllm_server` before sending requests.

    Model files resolve through the standard HF hub cache of the
    inherited environment ($HF_HOME/hub). Deliberately NO --download-dir
    override: vLLM resolves the config/tokenizer through the default hub
    cache regardless, so pointing only the weights elsewhere splits the
    cache and breaks HF_HUB_OFFLINE runs. Pre-cache the model under the
    default layout (plain `snapshot_download(model_id)`) on a node with
    network access.

    The subprocess inherits the environment with CUDA_VISIBLE_DEVICES
    overridden to the single target GPU, so vLLM's "GPU 0" is that GPU
    and the parent's other devices are invisible to it. stdout/stderr go
    to `log_path` (appended caller-side diagnostics belong there too) or
    are discarded if None. The process is registered with atexit so a
    normal interpreter exit tears it down; SLURM kills it with the job
    cgroup regardless.
    """
    port = _free_port()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = _visible_device_for(device)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", str(float(gpu_memory_utilization)),
    ]
    # PARITY: pin the serving dtype to bfloat16 so vLLM's forward pass matches
    # the HF policy load (plurpo.models.load_model uses bf16). Leaving it on
    # vLLM's "auto" can pick fp16 and shift logprobs by ~0.1+ nats vs HF.
    if dtype:
        cmd += ["--dtype", str(dtype)]
    if max_model_len:
        cmd += ["--max-model-len", str(int(max_model_len))]
    if enable_lora:
        # Serve the frozen base AND a hot-swappable LoRA on one engine: a
        # request with model=<base model_id> uses the base (the frozen sim);
        # model=<lora_name> uses base+adapter (the policy). max_lora_rank must
        # be >= the adapter's r (16 here); the adapter's alpha/scaling is read
        # from its adapter_config.json, matching PEFT.
        cmd += ["--enable-lora", "--max-lora-rank", str(int(max_lora_rank))]
        # Allow POST /v1/load_lora_adapter at runtime (per-iteration swap).
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"

    log_file = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    base_url = f"http://127.0.0.1:{port}"
    return proc, base_url


def wait_for_vllm_server(proc, base_url, log_path=None, timeout=1800):
    """Block until `base_url`/health answers 200. Polls via `curl -sf`
    every 2s (no try/except needed for the expected connection-refused
    phase). Fails loud if the server subprocess exits or `timeout`
    seconds pass — a silently absent judge would stall the whole run.
    """
    deadline = time.time() + timeout
    while True:
        assert proc.poll() is None, (
            f"vLLM server exited with code {proc.returncode} during startup; "
            f"see {log_path or 'its log'}"
        )
        ok = subprocess.run(
            ["curl", "-sf", "-o", "/dev/null", base_url + "/health"],
            check=False,
        ).returncode == 0
        if ok:
            return
        assert time.time() < deadline, (
            f"vLLM server at {base_url} not healthy after {timeout}s; "
            f"see {log_path or 'its log'}"
        )
        time.sleep(2)


_ENABLE_THINKING_OFF = {"enable_thinking": False}


def chat_template_kwargs_for(model_id, cache_dir=None):
    """Return the `chat_template_kwargs` to send for `model_id`, or None.

    `enable_thinking=False` exists to make Qwen3's chat template render the way
    `plurpo.models.build_chat_input` does. Some servers REJECT the field outright:
    vLLM auto-selects its Mistral tokenizer for Mistral-format repos, and
    `vllm/tokenizers/mistral.py:validate_request_params` raises
    "chat_template is not supported for Mistral tokenizers" whenever
    chat_template_kwargs is not None -- a 400 on every request, i.e. a run that
    dies on its first sample.

    Rule: send it only when the model's chat template actually references the
    variable. Verified 2026-07-28 by rendering each template both ways:

      Qwen3-8B        references it, renders DIFFERENTLY  -> must send
      gemma-4-12B-it  references it, renders identically  -> sent, no-op
      Ministral-3-14B does not reference it               -> must not send

    so every pre-existing arm keeps sending exactly what it sent before.

    Assumption: the tokenizer resolves from `cache_dir` / $HF_HOME offline,
    which is already true wherever the weights are being served from.
    """
    # Imported lazily so this module stays importable for the server-launch
    # helpers alone, without pulling in transformers.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    template = tokenizer.chat_template or ""
    if "enable_thinking" not in template:
        return None
    return dict(_ENABLE_THINKING_OFF)


def _extra_body(chat_template_kwargs):
    """extra_body for a chat request, omitting chat_template_kwargs entirely
    when it is None (sending null still trips the Mistral validator)."""
    if chat_template_kwargs is None:
        return {}
    return {"chat_template_kwargs": chat_template_kwargs}


def make_vllm_call_llm(base_url, model_id, request_timeout=600,
                       chat_template_kwargs=_ENABLE_THINKING_OFF):
    """Return a harness-signature `call_llm` routed to the vLLM server.

    Decoding is greedy (temperature=0), matching the HF sim path
    (`make_call_llm(do_sample=False)`); `enable_thinking=False` is
    forwarded via `chat_template_kwargs` so Qwen3's chat template
    renders identically to `plurpo.models.build_chat_input`. Token counts
    come from the server's usage block and are exact.

    The closure is thread-safe (the OpenAI client is); concurrent calls
    are batched server-side by vLLM's continuous batching — that is the
    entire point of this backend.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url + "/v1",
        api_key="EMPTY",
        timeout=request_timeout,
    )

    def call_llm(client_unused, model_name, prompt, max_tokens, system=None):
        del client_unused, model_name  # one local server; provider args ignored.
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=0.0,
            extra_body=_extra_body(chat_template_kwargs),
        )
        content = resp.choices[0].message.content
        assert content is not None, (
            f"vLLM returned no content (finish_reason="
            f"{resp.choices[0].finish_reason!r}) for prompt: {prompt[:200]!r}"
        )
        usage = resp.usage
        return content.strip(), int(usage.prompt_tokens), int(usage.completion_tokens)

    return call_llm


def load_lora_adapter(base_url, lora_name, lora_path, timeout=300):
    """Hot-load (or replace) a LoRA adapter on a running --enable-lora server
    via POST /v1/load_lora_adapter. After this returns, requests with
    model=`lora_name` apply that adapter on top of the frozen base. Idempotent
    re-load of the same name is how the trainer swaps in adapter_iter_k each
    iteration. Fails loud on a non-2xx so a silently-stale adapter can't
    corrupt the next round's samples."""
    import json as _json
    import urllib.request

    body = _json.dumps({"lora_name": lora_name, "lora_path": lora_path}).encode()
    req = urllib.request.Request(
        base_url + "/v1/load_lora_adapter", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        assert 200 <= r.status < 300, f"load_lora_adapter {lora_name} -> HTTP {r.status}"


def unload_lora_adapter(base_url, lora_name, timeout=120):
    """Unload `lora_name` if present (POST /v1/unload_lora_adapter). Tolerates a
    non-2xx (e.g. the adapter was never loaded), since callers unload-then-load
    to replace an adapter and the first round has nothing to unload."""
    import json as _json
    import urllib.error
    import urllib.request

    body = _json.dumps({"lora_name": lora_name}).encode()
    req = urllib.request.Request(
        base_url + "/v1/unload_lora_adapter", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=timeout).close()
    except urllib.error.HTTPError:
        pass  # not loaded yet — fine


def make_vllm_policy_sampler(base_url, model_id, request_timeout=1200,
                             chat_template_kwargs=_ENABLE_THINKING_OFF):
    """Return a sampler for the POLICY routed to the same vLLM engine.

    sample(prompt, n, temperature, top_p, max_tokens, adapter=None, greedy=False)
        -> list[str] of n decoded completions.

    `adapter` selects the LoRA by name (the current policy); None / "" uses the
    frozen base. PARITY: uses the SAME chat path as the HF policy sampler —
    chat.completions with `enable_thinking=False` so Qwen3's template renders
    identically to plurpo.models.build_chat_input. `greedy=True` forces
    temperature=0 (matches _policy_generate_greedy do_sample=False)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=request_timeout)

    def sample(prompt, n, temperature, top_p, max_tokens,
               adapter=None, greedy=False):
        if n <= 0:
            return []
        kw = dict(
            model=(adapter if adapter else model_id),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            n=int(n),
            extra_body=_extra_body(chat_template_kwargs),
        )
        if greedy:
            kw["temperature"] = 0.0
        else:
            kw["temperature"] = float(temperature)
            kw["top_p"] = float(top_p)
        resp = client.chat.completions.create(**kw)
        out = []
        for ch in resp.choices:
            assert ch.message.content is not None, (
                f"vLLM policy sampler empty content (finish={ch.finish_reason!r})"
            )
            out.append(ch.message.content.strip())
        return out

    return sample
