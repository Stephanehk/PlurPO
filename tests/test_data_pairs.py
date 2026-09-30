"""Integrity checks for the released preference-pair files in data/.

Asserts each file's row count, schema, and that the release clean filter holds:
no error-sentinel prompt and no prompt matching a held-out TEST prompt of its
env (whitespace/case-normalised). Counts are the ones documented in
data/README.md.

Run: pytest tests/test_data_pairs.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "train"))

from plurpo.io import read_jsonl
from rebuild_pairs import heldout_prompt_set, normalise

ENV_OF = {"oeq": "OEQ", "pas": "PAS", "aita": "AITA", "flip": "AITA-NTA-FLIP"}
COUNTS = {
    "plurpo_qwen3-8b_pairs": {"oeq": 27035, "pas": 37099, "aita": 21363, "flip": 16092},
    "qwen3-8b_role_confusion_pairs": {"oeq": 448, "pas": 2016, "aita": 328, "flip": 701},
}
SUFFIX = {"plurpo_qwen3-8b_pairs": ".jsonl.gz", "qwen3-8b_role_confusion_pairs": ".jsonl"}


def _check_dir(name, keys):
    for stem, n in COUNTS[name].items():
        pairs = read_jsonl(os.path.join(_ROOT, "data", name, f"{stem}_pairs{SUFFIX[name]}"))
        assert len(pairs) == n, (name, stem, len(pairs))
        heldout = heldout_prompt_set(ENV_OF[stem])
        for p in pairs:
            assert set(p) == keys, (name, stem, set(p))
            assert p["prompt"].strip().lower() != "error", (name, stem)
            assert normalise(p["prompt"]) not in heldout, (name, stem)


def test_stage1_pairs():
    _check_dir("plurpo_qwen3-8b_pairs", {"prompt", "chosen", "rejected", "epoch", "prompt_idx"})


def test_role_confusion_pairs():
    _check_dir("qwen3-8b_role_confusion_pairs", {"prompt", "chosen", "rejected"})
