"""Candidate-pool sampling from the current policy M1 (PlurPO step 2).

The k2 = n_samples candidates for a prompt come from three sources, always in
this order:
  "model"          n_model samples conditioned on the bare post;
  "perspshift"     n_steer samples conditioned on the ELEPHANT third-person
                   rewrite of the post (rewritten once, greedily, by the
                   current policy) -- or, for DPO-Neutral, "neutral" samples
                   conditioned on the post + NEUTRAL_SUFFIX;
  "stakeholder:X"  the remainder, round-robin over the prompt's panel, each
                   conditioned on STAKEHOLDER_RESPONSE_PROMPT for stakeholder X
                   (with 6 slots and 3 stakeholders, 2 each).
The generation prompts only diversify the pool: the DPO prompt column is always
the bare post.

`policy_sampler(prompt, n, temperature, top_p, max_tokens, adapter=None,
greedy=False) -> [text]` is `plurpo.vllm_server.make_vllm_policy_sampler`
serving base + the current LoRA under the adapter name "policy".
"""

from plurpo.perspective_shift import perspective_shift_post
from plurpo.prompts.baselines import NEUTRAL_SUFFIX
from plurpo.prompts.plurpo import STAKEHOLDER_RESPONSE_PROMPT

POLICY_ADAPTER = "policy"
STEER_SOURCES = ("perspshift", "neutral")


class CandidateSampler:
    """Draws one prompt's candidate pool.

    n_samples, n_model, n_steer: pool size and per-source counts; the rest go
    to stakeholders. steer: "perspshift" or "neutral" (ignored when
    n_steer == 0). perspshift_assess: append "Assess this narrative." to the
    rewrite. temperature/top_p/max_tokens: candidate decoding;
    rewrite_max_tokens: cap for the greedy rewrite.
    """

    def __init__(self, policy_sampler, n_samples=10, n_model=2, n_steer=2,
                 steer="perspshift", perspshift_assess=True, temperature=1.0,
                 top_p=0.95, max_tokens=1024, rewrite_max_tokens=1536):
        assert temperature > 0, "stochastic sampling is required for a diverse pool"
        assert n_model >= 0 and n_steer >= 0 and n_model + n_steer <= n_samples
        assert steer in STEER_SOURCES, f"unknown steer {steer!r}"
        self.policy_sampler = policy_sampler
        self.n_samples = n_samples
        self.n_model = n_model
        self.n_steer = n_steer
        self.steer = steer
        self.perspshift_assess = perspshift_assess
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.rewrite_max_tokens = rewrite_max_tokens

    @property
    def n_stakeholder(self):
        return self.n_samples - self.n_model - self.n_steer

    def _sample(self, prompt, n):
        if n <= 0:
            return []
        return self.policy_sampler(prompt, int(n), self.temperature, self.top_p,
                                   int(self.max_tokens), adapter=POLICY_ADAPTER)

    def _rewrite(self, prompt):
        return self.policy_sampler(prompt, 1, self.temperature, self.top_p,
                                   int(self.rewrite_max_tokens),
                                   adapter=POLICY_ADAPTER, greedy=True)[0]

    def sample(self, post, panel):
        """Return (responses, sources), parallel lists of length n_samples.
        `panel` is the post's [{"stakeholder", "concerns"}] list; it must be
        non-empty when there is a stakeholder budget."""
        responses, sources = [], []
        for r in self._sample(post, self.n_model):
            responses.append(r)
            sources.append("model")
        if self.n_steer > 0:
            if self.steer == "perspshift":
                steer_input = perspective_shift_post(self._rewrite, post,
                                                     self.perspshift_assess)
            else:
                steer_input = post + NEUTRAL_SUFFIX
            for r in self._sample(steer_input, self.n_steer):
                responses.append(r)
                sources.append(self.steer)
        if self.n_stakeholder > 0:
            assert panel, "stakeholder budget but no stakeholders for this prompt"
            counts = [0] * len(panel)
            for j in range(self.n_stakeholder):
                counts[j % len(panel)] += 1
            for entry, c in zip(panel, counts):
                if c == 0:
                    continue
                gen_prompt = STAKEHOLDER_RESPONSE_PROMPT.format(
                    stakeholder=entry["stakeholder"], post=post)
                for r in self._sample(gen_prompt, c):
                    responses.append(r)
                    sources.append(f"stakeholder:{entry['stakeholder']}")
        return responses, sources
