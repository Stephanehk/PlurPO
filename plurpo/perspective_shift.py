"""Perspective shift (Cheng et al., 2025, ELEPHANT, Sec. 4.3 + App. G.2).

Rewrite the user's first-person post into the third person, then answer the
rewrite instead of the user. Used in two places:
  - as the "Perspective-shift" prompting baseline at evaluation time, and
  - as one of PlurPO's three candidate sources during training (the rewrite is
    produced by the current policy, greedy; the answers to it are sampled).

Assumption: `rewrite_fn(prompt) -> text` is a single greedy call, so the
rewrite is deterministic and not itself a source of sampling noise. The
rewrite is passed through verbatim (stripped), including any residual "you",
which ELEPHANT documents as an expected failure mode of prompt-only rewriting.
"""

# Verbatim, ELEPHANT App. G.2 (primary rewrite instruction).
PERSPECTIVE_SHIFT_REWRITE_PROMPT = (
    'Rewrite this prompt into third-person. Changing the first person "I" to "someone".'
)

# Verbatim ELEPHANT App. G.2 addendum for long narratives, appended to the
# (already third-person) rewrite. Used for OEQ, AITA and AITA-NTA-FLIP; the
# PAS training runs did not use it (see configs/).
ASSESS_NARRATIVE_SUFFIX = "\n\nAssess this narrative."


def build_rewrite_prompt(post):
    """The user turn sent to the rewriter: the instruction, then the post."""
    return PERSPECTIVE_SHIFT_REWRITE_PROMPT + "\n\n" + post


def perspective_shift_post(rewrite_fn, post, assess_narrative):
    """Rewrite `post` into third person via `rewrite_fn` and return the text,
    optionally followed by ASSESS_NARRATIVE_SUFFIX. Asserts the rewrite is
    non-empty."""
    third_person = rewrite_fn(build_rewrite_prompt(post)).strip()
    assert third_person, "perspective-shift rewrite returned empty text"
    if assess_narrative:
        third_person = third_person + ASSESS_NARRATIVE_SUFFIX
    return third_person
