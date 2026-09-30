"""Single-call wrappers for each evaluation judge: build the verbatim prompt,
make the gpt-5-mini call with that judge's original settings, parse the label.

Per-judge call settings (all temperature 0, reasoning "minimal"), as used for
the paper:
  endorsement      max_tokens 4000, system ENDORSE_SYSTEM
  gate             max_tokens 4000, system GATE_SYSTEM, context/response
                   truncated to GATE_MAX_CHARS
  verdict          max_tokens 2000, system VERDICT_SYSTEM, response passed
                   through strip_verdict first
  role confusion   max_tokens 2000, system ROLE_CONFUSION_SYSTEM, fields
                   truncated to ROLE_CONFUSION_MAX_CHARS
  ELEPHANT         max_tokens 4000, system ELEPHANT_SYSTEM
"""

import os
import sys

from plurpo.judge_parsing import (first_endorse_label, parse_gate_label,
                                  parse_role_voice, parse_verdict, strip_verdict)
from plurpo.llm_api import chat
from plurpo.paths import VENDOR_DIR
from plurpo.prompts import judges as P

AITA_ENVS = ("AITA", "AITA-NTA-FLIP")


def endorse_prompt(version, sentence, response):
    """Fill the v1 or v5 endorsement template for one (question, response)."""
    return P.ENDORSE_TEMPLATES[version].format(question=sentence, advice=response)


def endorse_one(client, version, sentence, response):
    """-> (label 0-3 or None, chat result)."""
    r = chat(client, P.ENDORSE_SYSTEM, endorse_prompt(version, sentence, response), 4000)
    return first_endorse_label(r["content"]), r


def gate_prompt(env, context, response):
    """Fill the rejection-sampling gate prompt for `env`."""
    label, desc = P.GATE_FRAMING[env]
    return P.GATE_PROMPT.format(LABEL=label, DESC=desc,
                                context=str(context)[:P.GATE_MAX_CHARS],
                                response=str(response)[:P.GATE_MAX_CHARS])


def gate_one(client, env, context, response):
    """-> (label 0/1, chat result). An unparsable gate output is an assertion
    failure, never a defaulted label: a gate that invents labels would corrupt
    both the scored set and the rejection tally."""
    r = chat(client, P.GATE_SYSTEM, gate_prompt(env, context, response), 4000)
    label = parse_gate_label(r["content"].strip())
    assert label is not None, (
        f"gate judge returned no 0/1 label; raw output {r['content']!r}")
    return label, r


def verdict_prompt(post, response):
    """Fill the AITA verdict classifier prompt (response already stripped)."""
    return P.VERDICT_TEMPLATE.format(post=post, response=response)


def verdict_one(client, post, response):
    """-> (YTA/NTA/UNCLEAR or None, chat result). Strips the response's own
    trailing verdict cue first."""
    r = chat(client, P.VERDICT_SYSTEM, verdict_prompt(post, strip_verdict(response)), 2000)
    return parse_verdict(r["content"]), r


def role_confusion_prompt(env, context, response):
    """Fill the role-confusion (narrative voice) evaluation prompt."""
    is_post = env in AITA_ENVS
    framing = P.ROLE_CONFUSION_FRAMING_POST if is_post else P.ROLE_CONFUSION_FRAMING_SENTENCE
    body = (P.ROLE_CONFUSION_BODY_POST if is_post else P.ROLE_CONFUSION_BODY_SENTENCE).format(
        context=str(context)[:P.ROLE_CONFUSION_MAX_CHARS],
        response=str(response)[:P.ROLE_CONFUSION_MAX_CHARS])
    return P.ROLE_CONFUSION_TEMPLATE.format(framing=framing, body=body)


def role_confusion_one(client, env, context, response):
    """-> (STAKEHOLDER/OUTSIDER/UNCLEAR or "PARSE_FAIL", chat result)."""
    r = chat(client, P.ROLE_CONFUSION_SYSTEM, role_confusion_prompt(env, context, response), 2000)
    return parse_role_voice(r["content"]), r


def elephant_create_prompt():
    """Import the vendored ELEPHANT `create_prompt` (verbatim paper prompts).
    That module builds an OpenAI client at import time; a placeholder key lets
    it load -- the client is never used."""
    os.environ.setdefault("OPENAI_API_KEY", "unused-by-plurpo")
    sys.path.insert(0, os.path.join(VENDOR_DIR, "elephant"))
    from sycophancy_scorers import create_prompt
    return create_prompt


def elephant_one(client, create_prompt, metric, sentence, response):
    """-> (raw content, chat result) for ELEPHANT `metric` (validation /
    framing). The raw digit is parsed by the caller (0/1)."""
    prompt = create_prompt({"sentence": sentence, "response": response}, metric,
                           prompt_col="sentence", response_col="response")
    r = chat(client, P.ELEPHANT_SYSTEM, prompt, 4000)
    return r["content"].strip(), r
