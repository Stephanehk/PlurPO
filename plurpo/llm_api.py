"""OpenRouter client for the gpt-5-mini evaluation judges.

Every evaluation judge in the paper is `openai/gpt-5-mini` served through
OpenRouter's OpenAI-compatible API, called with a system + user message,
temperature 0 and reasoning effort "minimal" (so the model emits its label
cheaply instead of spending the budget on hidden reasoning). This module is the
single place that call is made.

Error policy: the client retries transient failures itself (`max_retries=12`);
anything that still raises is systematic (auth, quota, a malformed request)
and propagates. Every judge script is resumable, so a crash loses no finished
work. Needs OPENROUTER_API_KEY.
"""

import os
import threading

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "openai/gpt-5-mini"
# USD per 1M tokens for gpt-5-mini on OpenRouter, used only by the spend guard.
PRICE_IN = 0.25
PRICE_OUT = 2.00


def make_client(max_retries=12):
    """OpenAI-compatible client pointed at OpenRouter. Thread-safe."""
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    assert key, "OPENROUTER_API_KEY must be set to run the LM judges"
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=max_retries)


def chat(client, system, user, max_tokens, model=JUDGE_MODEL, temperature=0.0,
         reasoning="minimal"):
    """One judge call. Returns a dict with keys content (str, "" if the API
    returned no content), prompt_tokens, completion_tokens, reasoning_tokens,
    served_model. `temperature=None` / `reasoning=None` omit the field."""
    kwargs = dict(model=model, max_tokens=int(max_tokens),
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}])
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning is not None:
        kwargs["extra_body"] = {"reasoning": {"effort": reasoning}}
    resp = client.chat.completions.create(**kwargs)
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "content": resp.choices[0].message.content or "",
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        "served_model": getattr(resp, "model", model),
    }


class SpendGuard:
    """Thread-safe running USD total with an optional hard cap.

    `add(result)` accumulates one `chat` result and returns True once the cap
    (if any) has been reached, so callers can checkpoint and stop.
    """

    def __init__(self, max_cost=None, price_in=PRICE_IN, price_out=PRICE_OUT):
        self.max_cost = max_cost
        self.price_in = price_in
        self.price_out = price_out
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, result):
        with self._lock:
            self.spent += (result["prompt_tokens"] * self.price_in
                           + result["completion_tokens"] * self.price_out) / 1e6
            return self.max_cost is not None and self.spent >= self.max_cost
