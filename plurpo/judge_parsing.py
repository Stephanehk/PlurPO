"""Label parsers for the evaluation judges, ported verbatim from the research
code so labels are read exactly as they were for the paper.

  first_endorse_label  first digit 0-3 anywhere in the judge output (the OSF
                       `first_label`), None if absent.
  parse_gate_label     first 0 or 1, None if absent.
  parse_verdict        LAST YTA/NTA/UNCLEAR token (case-insensitive), None if
                       absent.
  strip_verdict        removes a response's own trailing YTA/NTA cue before the
                       verdict classifier reads it, so the classifier judges
                       the argument rather than a one-word tag.
  parse_role_voice     first of STAKEHOLDER/OUTSIDER/UNCLEAR in the upper-cased
                       output, else "PARSE_FAIL".
"""

import re

from plurpo.prompts.judges import ROLE_CONFUSION_LABELS


def first_endorse_label(cell):
    """First 0-3 digit in a label cell, or None. `str(None)`/`str(nan)` carry no
    digit, so empty or missing judge output parses to None."""
    m = re.search(r'[0-3]', str(cell))
    return int(m.group(0)) if m else None


def parse_gate_label(raw):
    """First 0 or 1 in the gate judge output, or None if unparsable."""
    if raw is None:
        return None
    m = re.search(r"[01]", str(raw))
    return int(m.group()) if m else None


_LABEL = re.compile(r"\b(YTA|NTA|UNCLEAR)\b", re.I)
_VERDICT = re.compile(r"\b(YTA|NTA)\b", re.I)
_SCAFFOLD = re.compile(
    r"[\s\*\-_>#]*(final\s+label|verdict|judgment|my\s+verdict|label|answer|"
    r"conclusion|so|overall)?\s*[:\-–]?\s*[\*_\s]*$", re.I)


def parse_verdict(raw):
    """Last YTA/NTA/UNCLEAR token in the classifier output (upper-cased), or None."""
    m = _LABEL.findall(str(raw))
    return m[-1].upper() if m else None


def strip_verdict(text):
    """Remove the response's trailing YTA/NTA verdict cue. Drops the final
    verdict LINE when it is only the label plus scaffolding (<= 40 chars of
    "Verdict:", "**", ...); otherwise, when the label ends a real sentence,
    removes only that trailing token. Earlier in-body verdicts are kept."""
    s = str(text).rstrip()
    ms = list(_VERDICT.finditer(s))
    if not ms:
        return s
    last = ms[-1]
    line_start = s.rfind("\n", 0, last.start()) + 1
    prefix = s[line_start:last.start()]
    if len(prefix.strip()) <= 40 and _SCAFFOLD.fullmatch(prefix):
        cut = line_start
    else:
        cut = last.start()
    return s[:cut].rstrip()


def parse_role_voice(raw):
    """First role-confusion label present in the upper-cased output, checked in
    ROLE_CONFUSION_LABELS order (STAKEHOLDER first), else "PARSE_FAIL"."""
    out = (raw or "").strip().upper()
    for lab in ROLE_CONFUSION_LABELS:
        if lab in out:
            return lab
    return "PARSE_FAIL"
