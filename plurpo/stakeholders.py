"""The simulated-stakeholder panel (PlurPO steps 1 and 3, Sec. 3.1).

A frozen copy of the model (M2), served greedily by vLLM, drives every call:

  panel     identify candidate stakeholders -> filter to at most k1 = 3 ->
            elicit each kept stakeholder's concerns. Policy-independent, so it
            is computed once per prompt and cached across epochs.
  veto      Stage 1 classifies each candidate's stance toward the narrator
            (ENDORSES / CHALLENGES) once. Stage 2 asks every panel member, plus
            optionally the user (narrator) stakeholder, to VETO or ACCEPT the
            candidate given that stance. A candidate is vetoed iff at least one
            voter vetoes it.

Unreadable judgments are never guessed. A candidate whose stance has no YES/NO
token, or that has no clean VETO but at least one vote with no VETO/ACCEPT
token, is EXCLUDED: it becomes neither a chosen nor a rejected response.

`call_llm(prompt, max_tokens) -> text` is any greedy, thread-safe completion
function over M2 (see `make_sim_call`). All fan-outs go through
`concurrent_map`, whose results are order-preserving, so the output does not
depend on the concurrency level for a deterministic judge.
"""

import re

from plurpo.concurrency import concurrent_map
from plurpo.prompts import plurpo as P

# The identify step is asked for fewer than MAX_STAKEHOLDERS; a model that
# over-fills is truncated here rather than aborting a multi-hour run. The
# filter step then keeps at most 3.
PARSE_MAX_STAKEHOLDERS = 15


def make_sim_call(vllm_call_llm):
    """Adapt a `plurpo.vllm_server.make_vllm_call_llm` closure to the
    `call_llm(prompt, max_tokens) -> text` shape used here."""
    def call_llm(prompt, max_tokens):
        text, _in, _out = vllm_call_llm(None, None, prompt, max_tokens)
        return text
    return call_llm


def parse_stakeholders(text):
    """One stakeholder per non-empty line, with bullet/number prefixes and
    markdown emphasis stripped; truncated at PARSE_MAX_STAKEHOLDERS."""
    items = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        s = re.sub(r"^(?:[\-\*•]|\d+[\.\)])\s*", "", s).strip()
        s = s.strip("*_`").strip()
        if s:
            items.append(s)
    return items[:PARSE_MAX_STAKEHOLDERS]


def parse_concerns(text):
    """Bullets under a `CONCERNS:` header (or every bullet if the header is
    missing), dropping "(none)" placeholders."""
    s = (text or "").strip()
    m = re.search(r"^\s*CONCERNS\s*:\s*\n(.*)\Z", s, re.I | re.MULTILINE | re.DOTALL)
    block = m.group(1) if m else s
    items = []
    for raw in block.splitlines():
        line = re.sub(r"^[\-\*•]\s*", "", raw.strip()).strip()
        if line and line.lower() not in ("(none)", "none"):
            items.append(line)
    return items


def format_bullets(items):
    """Concern list as "- item" lines, or "- (none)" when empty."""
    if not items:
        return "- (none)"
    return "\n".join(f"- {s}" for s in items)


def parse_stance(text):
    """Stage-1 reply -> (endorses, justification), or None if no YES/NO token.
    The justification is the whitespace-collapsed text after the token."""
    m = re.search(r"\b(YES|NO)\b", text, re.IGNORECASE)
    if m is None:
        return None
    return m.group(1).upper() == "YES", " ".join(text[m.end():].split())


def parse_vote(text):
    """Stage-2 reply -> True (VETO), False (ACCEPT), or None (unparsable)."""
    m = re.search(r"\b(VETO|ACCEPT)\b", text, re.IGNORECASE)
    if m is None:
        return None
    return m.group(1).upper() == "VETO"


class StakeholderSimulator:
    """Frozen stakeholder panel + veto judge over M2.

    veto_prompt: key of plurpo.prompts.plurpo.VETO_PROMPTS ("P6" for Qwen3 and
        Phi-4; per-family variants for Llama-3.1-8B and Granite-4.1-8B).
    user_stakeholder: add the narrator as an extra voter (never a candidate
        source).
    stakeholder_max_tokens / veto_max_tokens: generation caps for the panel
        calls and for the stance/veto calls.
    """

    def __init__(self, call_llm, veto_prompt="P6", user_stakeholder=True,
                 stakeholder_max_tokens=1024, veto_max_tokens=128):
        assert veto_prompt in P.VETO_PROMPTS, f"unknown veto prompt {veto_prompt!r}"
        self.call_llm = call_llm
        self.veto_template = P.VETO_PROMPTS[veto_prompt]
        self.user_stakeholder = user_stakeholder
        self.stakeholder_max_tokens = stakeholder_max_tokens
        self.veto_max_tokens = veto_max_tokens
        self._panels = {}
        self.n_excluded = 0

    # ---- step 1: the panel ------------------------------------------------

    def _identify_and_filter(self, post):
        """Candidate list from STAKEHOLDER_PROMPT, down-selected by FILTER_PROMPT.
        Asserts the filter kept at least one stakeholder."""
        text = self.call_llm(P.STAKEHOLDER_PROMPT.format(
            post=post, min_stakeholders=P.MIN_STAKEHOLDERS,
            max_stakeholders=P.MAX_STAKEHOLDERS), self.stakeholder_max_tokens)
        candidates = parse_stakeholders(text)
        text = self.call_llm(P.FILTER_PROMPT.format(
            post=post, stakeholders="\n".join(candidates)), self.stakeholder_max_tokens)
        kept = parse_stakeholders(text)
        assert kept, f"filter step removed all stakeholders. raw={text!r} candidates={candidates!r}"
        return kept

    def _concerns(self, task):
        """One stakeholder's concerns, retried once with CONCERNS_RETRY_SUFFIX
        when the first reply parses to none."""
        post, stakeholder = task
        prompt = P.CONCERNS_PROMPT.format(stakeholder=stakeholder, post=post)
        concerns = parse_concerns(self.call_llm(prompt, self.stakeholder_max_tokens))
        if not concerns:
            retry = self.call_llm(prompt + P.CONCERNS_RETRY_SUFFIX, self.stakeholder_max_tokens)
            concerns = parse_concerns(retry)
        return concerns

    def prime_panels(self, posts, concurrency):
        """Compute and cache the panel for every uncached post, breadth-first
        (all identify+filter calls, then all concerns calls)."""
        todo = [p for p in dict.fromkeys(posts) if p not in self._panels]
        if not todo:
            return
        kept = concurrent_map(self._identify_and_filter, todo, concurrency)
        tasks = [(post, s) for post, names in zip(todo, kept) for s in names]
        concerns = concurrent_map(self._concerns, tasks, concurrency)
        for post in todo:
            self._panels[post] = []
        for (post, s), c in zip(tasks, concerns):
            self._panels[post].append({"stakeholder": s, "concerns": c})

    def panel(self, post):
        """Cached [{"stakeholder", "concerns"}, ...] for a primed post."""
        return self._panels[post]

    # ---- step 3: vetoes ---------------------------------------------------

    def _stance(self, task):
        post, response = task
        text = self.call_llm(P.ENDORSE_CLASSIFY_PROMPT.format(post=post, response=response),
                             self.veto_max_tokens)
        return parse_stance(text)

    def _vote(self, task):
        post, response, entry, endorses, justification = task
        verdict = "ENDORSES" if endorses else "CHALLENGES"
        if entry is None:
            prompt = P.VETO_PROMPT_USER.format(
                post=post, response=response, stance_verdict=verdict,
                stance_justification=justification)
        else:
            prompt = self.veto_template.format(
                stakeholder=entry["stakeholder"], post=post,
                concerns_block=format_bullets(entry["concerns"]), response=response,
                stance_verdict=verdict, stance_justification=justification)
        return parse_vote(self.call_llm(prompt, self.veto_max_tokens))

    def veto_batch(self, items, concurrency):
        """Veto every candidate of every item.

        items: [(post, responses), ...] with primed panels.
        Returns (vetoed, excluded): per-item lists of responses, each in
        original response order. Indexing is by position, so duplicate texts
        keep per-occurrence semantics.
        """
        stance_tasks = [(i, j, post, r) for i, (post, rs) in enumerate(items)
                        for j, r in enumerate(rs)]
        stances = concurrent_map(lambda t: self._stance((t[2], t[3])), stance_tasks,
                                 concurrency)
        vote_tasks = []
        unknown = set()      # (i, j) with an unreadable stance or vote
        for (i, j, post, r), stance in zip(stance_tasks, stances):
            if stance is None:
                unknown.add((i, j))
                continue
            endorses, justification = stance
            voters = list(self._panels[post]) + ([None] if self.user_stakeholder else [])
            for entry in voters:
                vote_tasks.append((i, j, (post, r, entry, endorses, justification)))
        votes = concurrent_map(lambda t: self._vote(t[2]), vote_tasks, concurrency)

        vetoed_pos = set()
        for (i, j, _task), vote in zip(vote_tasks, votes):
            if vote is None:
                unknown.add((i, j))
            elif vote:
                vetoed_pos.add((i, j))
        vetoed, excluded = [], []
        for i, (_post, rs) in enumerate(items):
            vetoed.append([r for j, r in enumerate(rs) if (i, j) in vetoed_pos])
            excluded.append([r for j, r in enumerate(rs)
                             if (i, j) in unknown and (i, j) not in vetoed_pos])
        self.n_excluded += sum(len(e) for e in excluded)
        return vetoed, excluded

    # ---- ablation: pairwise preference labels instead of vetoes ------------

    def prefer(self, post, response_a, response_b, max_tokens=128):
        """PlurPO-no-stakeholder-preferences labeler: "A", "B", or None when
        the frozen model's reply has no A/B token."""
        text = self.call_llm(P.PREFERENCE_PROMPT.format(
            post=post, response_a=response_a, response_b=response_b), max_tokens)
        m = re.search(r"\b([AB])\b", (text or "").strip())
        return None if m is None else m.group(1).upper()
