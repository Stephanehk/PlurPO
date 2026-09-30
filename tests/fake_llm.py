"""Deterministic fake LLM + fake policy sampler for pipeline tests.

Replies are pure functions of the prompt text (sha256), so any code path that
sends byte-identical prompts gets byte-identical replies regardless of call
order or thread scheduling. Every call is recorded as (prompt, max_tokens) so
tests can compare the exact prompts two implementations send.
"""

import hashlib
import threading


def _h(text, salt=""):
    return int(hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:12], 16)


class FakeLLM:
    """call(prompt, max_tokens) -> text, dispatching on recognisable
    fragments of the PlurPO prompt templates."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, prompt, max_tokens):
        with self.lock:
            self.calls.append((prompt, int(max_tokens)))
        h = _h(prompt)
        if "distinct stakeholders whose perspectives" in prompt:
            n = 2 + h % 17   # sometimes over-fills past the parse cap (15)
            styles = ["{i}. Person {i}", "- **Person {i}**", "Person {i}", "* Person {i}"]
            return "\n".join(styles[(h + i) % 4].format(i=i) for i in range(n))
        if "You previously identified the following stakeholders" in prompt:
            names = [l for l in prompt.split("stakeholders for the situation above:\n\n")[1]
                     .split("\n\nSome of these")[0].split("\n") if l.strip()]
            return "\n".join(names[: 1 + h % 3])
        if "list:\n  CONCERNS:" in prompt:
            if h % 4 == 0:
                return "CONCERNS:\n- (none)"
            return "CONCERNS:\n" + "\n".join(f"- (action: x{j}) concern {h % 97}-{j}"
                                           for j in range(1 + h % 3))
        if "Does the response endorse the narrator's own action" in prompt:
            r = h % 10
            if r == 0:
                return "Hmm, hard to say."
            return ("YES" if r < 6 else "NO") + f"\nbecause reason {h % 1000}   with  spaces"
        if "Which response is better?" in prompt:
            r = h % 7
            return "neither" if r == 0 else ("A" if r < 4 else "B") + "\nreason"
        # stance-conditioned veto prompts (stakeholders and the user)
        r = h % 9
        if r == 0:
            return "CHALLENGE, I think."
        return ("VETO" if r < 4 else "ACCEPT") + "\nshort reason"


class FakePolicy:
    """policy_sampler(prompt, n, temperature, top_p, max_tokens, adapter=None,
    greedy=False) -> list of n texts; greedy calls return one rewrite."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, prompt, n, temperature, top_p, max_tokens, adapter=None, greedy=False):
        with self.lock:
            self.calls.append((prompt, int(n), float(temperature), float(top_p),
                               int(max_tokens), adapter, bool(greedy)))
        if greedy:
            return [f"Someone wrote a third-person version #{_h(prompt) % 1000}."]
        h = _h(prompt)
        # a few duplicate texts on purpose, to exercise per-occurrence semantics
        return [f"response {h % 50}-{(k * 7) % max(1, n - 1)}" for k in range(n)]
