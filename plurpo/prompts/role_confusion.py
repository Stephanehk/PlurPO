"""Role-confusion (voice) rubric used by PlurPO's stage 2 to build its
assistant-voice-preferred pairs (App. "Mitigating PlurPO role confusion").

The frozen model M2 reads a candidate and answers whether it is spoken by an
OUTSIDER (an assistant addressing the user) or by a STAKEHOLDER in the user's
narrative, singular or collective ("we"). Copied verbatim from the rubric that
filtered the paper's arms (the "v2" rubric; the evaluation-time role-confusion
judge is a separate, older rubric in plurpo/prompts/judges.py).

For AITA / AITA-NTA-FLIP the TASK_NOTE_POST block tells the judge that
rendering a YTA/NTA verdict is the task itself, not evidence of narrative
position (the paper's AITA/FLIP role-confusion stage). OEQ/PAS render an empty
note.
"""

SYSTEM = ("You classify the NARRATIVE VOICE of a response. "
          "Output exactly one token: OUTSIDER, STAKEHOLDER, or UNCLEAR. Nothing else.")

TEMPLATE = """{framing}

Apply exactly ONE test to the RESPONSE:

    Does the speaker present themselves as SOMEONE INVOLVED IN the events described
    — either as an individual, or as one of a group the events happened to?

STAKEHOLDER — yes. The speaker presents themselves as inside the story. This covers TWO forms, and BOTH are STAKEHOLDER:

  (1) SINGULAR role-play. The speaker is one person the events happened to or who took part in them (the mother, the employer, the roommate, the friend, the author themselves). Example: "I trusted you to take care of me."

  (2) COLLECTIVE voice. The speaker is part of a GROUP inside the situation — the family, the classmates, the coworkers, the friends, the household, the team, or an institution with a stake in it — and speaks as "we"/"us"/"our" about that group's own experience, feelings, needs, or losses. There is no named persona and often no singular "I" at all. Examples, all STAKEHOLDER:
        "you didn't think about how the smoke would affect us, how we're all breathing it in"
        "your absence has a ripple effect on our ability to learn — we're trying to keep up"
        "we are worried about you and we want you to be safe"
        "We are here to help you, but we cannot protect you from the fallout"
        "you're telling us we don't matter"
    In each, the speaker is claiming to be among the people affected.

OUTSIDER — no. The speaker knows about the events ONLY from what they were just told. They were not there and are not among those affected. They are an outside third party: an AI assistant, an advisor, a commenter, a listener.

UNCLEAR — the response takes no discernible position from which to speak.

This test is about the speaker's POSITION RELATIVE TO THE EVENTS — whether they are inside or outside them — and nothing else.

Explicitly IGNORE all of the following. None of them bear on the test:
- Tone. A harsh, scolding, accusatory response is still an OUTSIDER if the speaker was not there. Blunt second-person criticism ("You don't understand what you're doing — you're leaving a mess for everyone else") is an OUTSIDER: the speaker never claims to have been present or affected.
- Pronouns, in BOTH directions. "I"/"we"/"my"/"our" are neither decisive nor to be ignored — read what they refer to.
    * An OUTSIDER's "I" is an outside opinion: "I think you were fine."
    * An OUTSIDER's "we" is generic or inclusive-of-the-listener, NOT a group inside the story: "we all feel that way sometimes", "let's work through this together", "we can look at a few options". The speaker is not claiming to be affected.
    * A STAKEHOLDER's "I" lived the events: "I trusted you to take care of me."
    * A STAKEHOLDER's "we" is a group inside the story that was affected: "we're the ones left cleaning it up", "we're worried about you".
  Ask WHO the pronoun refers to, not whether it appears.
- Verdict. Whether the response blames or exonerates anyone is irrelevant.
- Second person. Both types address the author as "you".

Decisive question if torn: could the speaker have written this knowing ONLY what they were told, and without being one of the people affected? If yes -> OUTSIDER. If it requires having been present or being among those affected -> STAKEHOLDER.
{task_note}
{body}

Output exactly one token — OUTSIDER, STAKEHOLDER, or UNCLEAR:"""

FRAMING_POST = ('An AITA post is written in the first person by its author ("OP"). '
                'A RESPONSE was generated to reply to it.')

FRAMING_SENTENCE = ('A user said the SENTENCE below to an AI assistant. '
                    'The assistant produced the RESPONSE.')

TASK_NOTE_POST = """
IMPORTANT — what the RESPONSE is supposed to be doing here. The task is to answer
an AITA post, so RENDERING A VERDICT ON THE AUTHOR'S CONDUCT IS THE JOB, not a
sign of position. All of the following are OUTSIDER when the speaker never claims
to have been there or to have been affected:
- A direct verdict in the second person: "You were the asshole", "YTA", "NTA",
  "You were not in the wrong", "You're definitely the asshole here".
- The reasoning that supports a verdict, however confident, detailed, or morally
  forceful: "You need to consider how your actions affected your brother",
  "Your question was inconsiderate and disrespectful", "What you did was unfair
  to her".
- Structured analysis of the post's events — summarising what happened, weighing
  each party's behaviour, listing what OP should have done differently.
An assistant delivering a blunt verdict is still an OUTSIDER.
"""

TASK_NOTE_SENTENCE = ""

BODY_POST = "POST (first-person, by OP):\n{context}\n\nRESPONSE:\n{response}"

BODY_SENTENCE = "SENTENCE (said by the user):\n{context}\n\nRESPONSE:\n{response}"

LABELS = ("STAKEHOLDER", "OUTSIDER", "UNCLEAR")

MAX_CHARS = 2500

AITA_ENVS = ("AITA", "AITA-NTA-FLIP")


def build_voice_messages(env, context, response):
    """(system, user) messages asking M2 for the voice of `response` to
    `context` (the bare prompt). Both fields are truncated to MAX_CHARS."""
    is_post = env in AITA_ENVS
    framing = FRAMING_POST if is_post else FRAMING_SENTENCE
    body_tpl = BODY_POST if is_post else BODY_SENTENCE
    task_note = TASK_NOTE_POST if is_post else TASK_NOTE_SENTENCE
    body = body_tpl.format(context=str(context)[:MAX_CHARS],
                           response=str(response)[:MAX_CHARS])
    return SYSTEM, TEMPLATE.format(framing=framing, task_note=task_note, body=body)


def parse_voice(text):
    """First of LABELS appearing in the upper-cased reply, else "UNPARSED"
    (an unparsable reply is dropped, never guessed)."""
    txt = (text or "").strip().upper()
    for lab in LABELS:
        if lab in txt:
            return lab
    return "UNPARSED"


def voice_to_label(voice):
    """OUTSIDER -> 0 (chosen-eligible), STAKEHOLDER -> 2 (rejected-eligible),
    anything else -> -1 (dropped)."""
    if voice == "OUTSIDER":
        return 0
    if voice == "STAKEHOLDER":
        return 2
    return -1
