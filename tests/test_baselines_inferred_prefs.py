"""Inferred-Prefs-DPO pair construction.

Synthetic checks of the rule (exact 4-way label match vs the human label;
truncated / unlabelled candidates dropped; per-prompt cap; determinism), plus a
byte-for-byte reproduction of the paper's OEQ and PAS pair files from their
research candidates + labels when those artifacts are present on this machine
(skipped otherwise)."""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "baselines"))

from inferred_prefs_pairs import (inferred_prefs_pairs, read_candidate_labels,
                                  read_human_labels)
from plurpo.io import read_keyed_candidates

# Optional: point PLURPO_RESEARCH_ROOT at a copy of the original research
# artifacts (runs/ and sycoharm_runs/) to enable the byte-for-byte check.
RESEARCH_ROOT = os.environ.get("PLURPO_RESEARCH_ROOT", "")
RESEARCH_RUNS = os.path.join(RESEARCH_ROOT, "runs")
RESEARCH_SR = os.path.join(RESEARCH_ROOT, "sycoharm_runs")


def _cands(spec):
    """spec: [(pid, idx, response, finished)] -> candidates dict."""
    return {(p, i): {"sentence": f"post {p}", "response": r, "finished": f}
            for p, i, r, f in spec}


def test_rule():
    c = _cands([("a", 0, "m1", True), ("a", 1, "x1", True), ("a", 2, "x2", True),
                ("a", 3, "t", False), ("b", 0, "only-match", True), ("c", 0, "u", True)])
    labels = {("a", 0): 1, ("a", 1): 0, ("a", 2): 3, ("a", 3): 0, ("b", 0): 2, ("c", 0): None}
    human = {"a": 1, "b": 2, "c": 0}
    pairs, man = inferred_prefs_pairs(c, labels, human, 10, 0)
    assert sorted(p["rejected"] for p in pairs) == ["x1", "x2"]
    assert all(p["chosen"] == "m1" and p["prompt"] == "post a" for p in pairs)
    assert man["n_truncated_dropped"] == 1
    assert man["n_prompts_dropped_no_mismatch"] == 2      # b (all match) and c (unlabelled)
    assert pairs == inferred_prefs_pairs(c, labels, human, 10, 0)[0]


def test_cap():
    c = _cands([("a", i, f"r{i}", True) for i in range(12)])
    labels = {("a", 0): 1, **{("a", i): 0 for i in range(1, 12)}}
    pairs, man = inferred_prefs_pairs(c, labels, {"a": 1}, 10, 0)
    assert len(pairs) == 10 and man["n_prompts_hit_pair_cap"] == 1


@pytest.mark.parametrize("env,human", [
    ("oeq", "human_endorse_oeq_origdata_train.csv"),
    ("pas", "assumed0_human_endorse_pas_origdata_train.csv"),
])
def test_reproduces_research_pairs(env, human):
    d = f"{RESEARCH_RUNS}/dpo_lora_{env}_qwen3-8b_n1000_origdata_inferredprefs_shared/pairs"
    if not RESEARCH_ROOT or not os.path.exists(f"{d}/pairs.jsonl"):
        pytest.skip("research artifacts not available")
    pairs, man = inferred_prefs_pairs(read_keyed_candidates(f"{d}/candidates.jsonl"),
                                      read_candidate_labels(f"{d}/labels.csv"),
                                      read_human_labels(f"{RESEARCH_SR}/{human}"), 10, 0)
    with open(f"{d}/pairs.jsonl") as f:
        assert "".join(json.dumps(p) + "\n" for p in pairs) == f.read()
    with open(f"{d}/manifest.json") as f:
        assert man == json.load(f)
