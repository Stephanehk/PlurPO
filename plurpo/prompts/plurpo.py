"""All prompts PlurPO sends to the frozen model M2 and the policy M1 during
training (Sec. 3.1 and App. "PlurPO Implementation Details").

Every template is copied verbatim from the code that produced the paper's
results; tests/test_prompts_frozen.py pins each one by sha256.

Step 1, synthesize stakeholders (M2, greedy):
  STAKEHOLDER_PROMPT   -> up to MAX_STAKEHOLDERS-1 candidate stakeholders
  FILTER_PROMPT        -> down-select to at most k1 = 3
  CONCERNS_PROMPT      -> each kept stakeholder's concerns (one retry with
                          CONCERNS_RETRY_SUFFIX if none parse)
Step 2, candidate generation (M1, sampled): STAKEHOLDER_RESPONSE_PROMPT
  conditions a candidate on one stakeholder's perspective (second person,
  addressed to the narrator).
Step 3, simulate vetoes (M2, greedy):
  ENDORSE_CLASSIFY_PROMPT -> does the candidate endorse or challenge the
                             narrator? (once per candidate)
  VETO_PROMPTS[name]      -> per-stakeholder VETO/ACCEPT given that stance
  VETO_PROMPT_USER        -> the additional user (narrator) stakeholder
The veto-prompt variants other than "P6" are the per-family adaptations of
App. "Training other models with PlurPO" (see configs/).

PREFERENCE_PROMPT is used only by the PlurPO-no-stakeholder-preferences
ablation (pairwise A/B labels instead of stakeholder vetoes).
"""

# Bounds substituted into STAKEHOLDER_PROMPT ("between 1 and fewer than 10").
MIN_STAKEHOLDERS = 1
MAX_STAKEHOLDERS = 10

STAKEHOLDER_PROMPT = """
---
Input you will write a response to:
{post}
---

List between {min_stakeholders} and fewer than {max_stakeholders} distinct stakeholders whose perspectives should be considered when evaluating the narrator's behavior. Exclude the narrator.

A stakeholder MUST be one of:
  - a specific person mentioned or directly implied in the input (e.g. a named partner, parent, friend, coworker), OR
  - a person who would be directly affected by the choice the narrator is making or the action they describe.

DO NOT list:
  - generic role categories that are not tied to the narrator's situation (e.g. "online forum moderator", "religious group leader", "tourist guide", "insurance company representative" on a question that doesn't involve any of those)
  - service providers, professionals, or institutions the narrator might hypothetically contact in the future ("homeless service provider" when the narrator hasn't sought one out, "skydiving instructor" when the question is about overcoming a fear, "career counselor" when the narrator isn't considering hiring one)
  - emotional states, perspectives, themes, or aspects of the narrator's well-being ("the narrator's loneliness", "the narrator's future self")
  - hypothetical future people whose existence depends on the narrator's choice (e.g. "potential future romantic partners" when the question is about the past)
  - people thay do not currently exist (e.g., someone's future self)
  - people that reasonably should have no bearing on the response to the input. 

If the input does not contain enough concrete people to fill {max_stakeholders} slots, output fewer rather than inventing speculative stakeholders.

Your list MAY include up to one neutral third-party observer who has no relationship to anyone involved — only if the input actually involves identifiable other people whose perspective is in tension.

Output one stakeholder per line, as a short noun phrase identifying a specific person or named role tied to this situation. Do not number them. Do not add any commentary, headers, or blank lines."""

FILTER_PROMPT = """
---
{post}
---

You previously identified the following stakeholders for the situation above:

{stakeholders}

Some of these stakeholders may not be relevant. Filter the list down to only those whose perspective should genuinely be taken into account when evaluating the narrator's behavior. Remove any stakeholder whose existence, involvement, or relevance is not established or strongly implied by the input itself — for example, hypothetical or counterfactual persons, authorities whose involvement requires inferring context the input does not establish, or near-duplicates of another stakeholder.

GUIDELINES:
- output AT MOST 3 stakeholders
- choose the 3 MOST IMPORTANT / MOST RELEVANT stakeholders
- remove redundant stakeholders
- remove stakeholders who are not directly relevant to the input

Where the input supports it, try to keep at least one neutral third-party observer who has no relationship to anyone involved, and at least one stakeholder whose interests are likely to conflict with how the narrator framed the situation, so that a critical perspective is not lost.

Output one stakeholder per line, as a short noun phrase. Do not number them. Do not add any commentary, headers, or blank lines."""

CONCERNS_PROMPT = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation is narrated from one party's perspective — the person whose conduct is being evaluated. Read the input critically but charitably: do not assume the narrator is misrepresenting events unless the input itself contains evidence of that.

From your perspective as {stakeholder}, list:
  CONCERNS: things the narrator did or said in the situation that genuinely affected you negatively, or that you would object to. THESE ARE CONCERNS ABOUT THE NARRATOR. Each concern MUST reference a specific thing the narrator described themselves doing, planning to do, or saying — begin each bullet with "(action: <quote or paraphrase from the input>)" for completed actions or statements, or "(plan: <quote or paraphrase>)" for stated intentions or things the narrator says they will do. Do not raise concerns about actions, attitudes, or motives the narrator did not actually describe. If you have no real concerns, write "- (none)" — do not manufacture grievances to fill the slot.

Reply in exactly this format and nothing else. Use one bullet per line, prefixed with "- ".

REPLY ONLY AS THE STAKEHOLDER YOU ARE ROLE-PLAYING WOULD. DO NOT CONSIDER WHAT ANYONE ELSE WOULD SAY. 

CONCERNS:
- (action: <quote or paraphrase>) <your concern>
- (plan: <quote or paraphrase>) <your concern>

The situation:
---
{post}
---"""

CONCERNS_RETRY_SUFFIX = "\n\nThe input is the full context/all that is provided by the user. Please respond as best you can assuming no more additional information can be provided."

STAKEHOLDER_RESPONSE_PROMPT = """Consider the perspective of the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

Write a response addressed directly to the narrator that reflects {stakeholder}'s perspective and concerns about the situation. Address the narrator directly as "you"; never reply in the voice of {stakeholder} or any other person described. Write only the response to the narrator and nothing else."""

ENDORSE_CLASSIFY_PROMPT = """Does the response endorse the narrator's own action, or challenge it?

Situation:
---
{post}
---

Response:
---
{response}
---

The narrator's "action" is what they did, decided, or plan to do, the goal they pursue, and how they frame the other people -- NOT their feelings.

- CHALLENGE (answer NO): the response questions, warns against, discourages, or criticizes the narrator's OWN action/choice/plan/goal in any way -- even briefly, gently, or alongside helpful advice.
- ENDORSE (answer YES): the response does NOT question the narrator's action, and instead says it was right/understandable, sides with the narrator against the other people, adopts the narrator's framing, or helps them carry out their goal.
- Validating ONLY feelings/character with no side taken and no help toward the goal: answer NO.

Criticism aimed at the OTHER people (not the narrator) is endorsement of the narrator, so answer YES.

Reply with a single token YES or NO on the first line, then a one-sentence justification:

YES or NO
<one-sentence justification>"""

# P6: the veto prompt used for Qwen3-8B/32B and Phi-4.
VETO_PROMPT_P6 = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

In an earlier review, you (as {stakeholder}) surfaced these concerns about the narrator's actions and statements in the situation:

CONCERNS:
{concerns_block}

Someone is considering sending the narrator the following response to their situation. Read it carefully:
---
{response}
---

A prior analysis determined that this response {stance_verdict} the narrator's OWN action (what they did, decided, or plan to do, the goal they pursue, and how they frame the other people). The reason given was:

"{stance_justification}"

Question: As {stakeholder}, given your concerns above, what do YOU think the narrator's own action deserves — to be ENDORSED (affirmed, sided with, encouraged, helped along) or CHALLENGED (questioned, warned against, discouraged, pushed back on)? This is about the narrator's action, NOT their feelings.

Then decide:
VETO if the response does NOT match your view — i.e. it ENDORSES the narrator when you would CHALLENGE them, OR it CHALLENGES the narrator when you would ENDORSE them.
ACCEPT if the response matches your view — or if, reading it yourself, you conclude the prior analysis mis-read its stance and it actually matches what you want.

Reply with a single token VETO or ACCEPT on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

VETO or ACCEPT
<one-sentence justification>"""

# Llama-3.1-8B adaptation.
VETO_PROMPT_P6_STANCE_ONLY = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

In an earlier review, you (as {stakeholder}) surfaced these concerns about the narrator's actions and statements in the situation:

CONCERNS:
{concerns_block}

Someone is considering sending the narrator the following response to their situation. Read it carefully:
---
{response}
---

A prior analysis determined that this response {stance_verdict} the narrator's OWN action (what they did, decided, or plan to do, the goal they pursue, and how they frame the other people). The reason given was:

"{stance_justification}"

Question: As {stakeholder}, given your concerns above, what do YOU think the narrator's own action deserves — to be ENDORSED (affirmed, sided with, encouraged, helped along) or CHALLENGED (questioned, warned against, discouraged, pushed back on)? This is about the narrator's action, NOT their feelings.

GUIDELINES:
- You are judging ONE thing: whether the response's stance on the narrator's action matches yours. Whether it also addresses your particular concern, goes far enough, or gives your perspective enough weight is NOT what is being asked, and is never on its own a reason to VETO.
- Take the prior analysis's reading of the stance as given unless the response's own words plainly contradict it. If the response questions, warns against, or pushes back on the narrator's action anywhere in it, then it CHALLENGES them — even if it does so gently, only in part, or alongside validating their feelings.
- Name the mismatch or ACCEPT. If the one-sentence justification you are about to write says the response's stance agrees with your view, the answer is ACCEPT, not VETO.

Then decide:
VETO if the response does NOT match your view — i.e. it ENDORSES the narrator when you would CHALLENGE them, OR it CHALLENGES the narrator when you would ENDORSE them.
ACCEPT if the response matches your view — or if, reading it yourself, you conclude the prior analysis mis-read its stance and it actually matches what you want.

Reply with a single token VETO or ACCEPT on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

VETO or ACCEPT
<one-sentence justification>"""

# Granite-4.1-8B adaptation.
VETO_PROMPT_P6_GRANITE = """You are role-playing as the following stakeholder in the situation described below: {stakeholder}.

The situation:
---
{post}
---

In an earlier review, you (as {stakeholder}) surfaced these concerns about the narrator's actions and statements in the situation:

CONCERNS:
{concerns_block}

Someone is considering sending the narrator the following response to their situation. Read it carefully:
---
{response}
---

A prior analysis determined that this response {stance_verdict} the narrator's OWN action (what they did, decided, or plan to do, the goal they pursue, and how they frame the other people). The reason given was:

"{stance_justification}"

Question: As {stakeholder}, given your concerns above, what do YOU think the narrator's own action deserves — to be ENDORSED (affirmed, sided with, encouraged, helped along) or CHALLENGED (questioned, warned against, discouraged, pushed back on)? This is about the narrator's action, NOT their feelings.

GUIDELINES:
- You are judging ONE thing: whether the response's stance on the narrator's action matches yours. Whether it also addresses your particular concern, goes far enough, or gives your perspective enough weight is NOT what is being asked, and is never on its own a reason to VETO.
- Name the mismatch or ACCEPT. If the one-sentence justification you are about to write says the response's stance agrees with your view, the answer is ACCEPT, not VETO.
- Your first line is the DECISION, never your own view. Work out your view silently, compare it to the response's stance, and write only VETO or ACCEPT there — never ENDORSE, ENDORSED, CHALLENGE or CHALLENGED.

Then decide:
VETO if the response does NOT match your view — i.e. it ENDORSES the narrator when you would CHALLENGE them, OR it CHALLENGES the narrator when you would ENDORSE them.
ACCEPT if the response matches your view — or if, reading it yourself, you conclude the prior analysis mis-read its stance and it actually matches what you want.

Reply with a single token VETO or ACCEPT on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

VETO or ACCEPT
<one-sentence justification>"""

# The user (narrator) stakeholder, added to the panel when enabled.
VETO_PROMPT_USER = """You are role-playing as the narrator — the person who wrote the situation below and is asking for a response. Answer honestly from their point of view.

Your situation, in your own words:
---
{post}
---

Someone is considering sending you the following response to your situation. Read it carefully:
---
{response}
---

A prior analysis determined that this response {stance_verdict} your OWN action (what you did, decided, or plan to do, the goal you pursue, and how you frame the other people). The reason given was:

"{stance_justification}"

Question: As the narrator, thinking honestly about your own situation, does your action deserve to be ENDORSED (affirmed, supported, encouraged) or CHALLENGED (questioned, warned against, pushed back on)? Be honest with yourself — if you genuinely acted wrongly, answer CHALLENGE; if you were reasonable, answer ENDORSE.

Then decide:
VETO if the response does NOT match your view — i.e. it CHALLENGES you when you deserve to be ENDORSED, OR it ENDORSES you when you deserve to be CHALLENGED.
ACCEPT if the response matches your view.

Reply with a single token VETO or ACCEPT on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

VETO or ACCEPT
<one-sentence justification>"""

VETO_PROMPTS = {
    "P6": VETO_PROMPT_P6,
    "P6_stance_only": VETO_PROMPT_P6_STANCE_ONLY,
    "P6_granite": VETO_PROMPT_P6_GRANITE,
}

PREFERENCE_PROMPT = """You are comparing two candidate responses to the situation below, to decide which is the better response for the narrator.

The situation:
---
{post}
---

Response A:
---
{response_a}
---

Response B:
---
{response_b}
---

Question: Which response is better?

Reply with a single token A or B on the first line, followed by a one-sentence justification on the second line. Use this exact format and nothing else:

A or B
<one-sentence justification>"""

