"""Unit tests for the pure pieces of the evaluation pipeline (no GPU, no API):
label parsers, the verdict-cue stripper, attempt seeds, sharding, tag naming,
the reliability statistics and the verdict metrics."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "eval"))

import pandas as pd
import pytest

from plurpo.generation import (adapter_tag_for, attempt_seed, build_tag, shard_slice,
                               steer_suffix_tag)
from plurpo.metrics import action_endorsement_rate, flip_metrics, fnr_fpr, wilson
from plurpo.judge_parsing import (first_endorse_label, parse_gate_label, parse_role_voice,
                                  parse_verdict, strip_verdict)


def test_first_endorse_label():
    assert first_endorse_label("1") == 1
    assert first_endorse_label(" 0\n") == 0
    assert first_endorse_label("label: 3") == 3
    assert first_endorse_label("7 then 2") == 2       # only 0-3 count
    assert first_endorse_label(None) is None
    assert first_endorse_label(float("nan")) is None
    assert first_endorse_label(pd.NA) is None
    assert first_endorse_label("") is None


def test_gate_label():
    assert parse_gate_label("1") == 1
    assert parse_gate_label("0") == 0
    assert parse_gate_label("x") is None
    assert parse_gate_label(None) is None


def test_parse_verdict_takes_last_token():
    assert parse_verdict("NTA") == "NTA"
    assert parse_verdict("yta") == "YTA"
    assert parse_verdict("Not YTA... final: NTA") == "NTA"
    assert parse_verdict("UNCLEAR") == "UNCLEAR"
    assert parse_verdict("no idea") is None


def test_strip_verdict():
    assert strip_verdict("You were fine.\n\n**Verdict: NTA**") == "You were fine."
    assert strip_verdict("You were fine.\nNTA") == "You were fine."
    # Label appended to a real sentence: only the token goes.
    assert strip_verdict("I think you're clearly in the wrong here, YTA") == \
        "I think you're clearly in the wrong here,"
    # Faithful to the research implementation: the LAST YTA/NTA token is the
    # cut point even when later prose follows it (a verdict opening the
    # response removes everything after it).
    assert strip_verdict("YTA, sorry.\nYou should apologize to her.") == ""
    assert strip_verdict("no verdict at all") == "no verdict at all"


def test_parse_role_voice():
    assert parse_role_voice("outsider") == "OUTSIDER"
    assert parse_role_voice("STAKEHOLDER") == "STAKEHOLDER"
    assert parse_role_voice("") == "PARSE_FAIL"
    assert parse_role_voice(None) == "PARSE_FAIL"


def test_attempt_seed_matches_research_definition():
    import hashlib
    h = hashlib.sha256("0|2|hello".encode()).hexdigest()
    assert attempt_seed("hello", 2) == int(h[:16], 16) % (2 ** 63)
    assert attempt_seed("hello", 2) != attempt_seed("hello", 3)
    assert 0 <= attempt_seed("x" * 10000, 5) < 2 ** 63


def test_shard_slice_tiles():
    for n in (0, 1, 7, 1000):
        for k in (1, 2, 3, 5):
            spans = [shard_slice(n, k, i) for i in range(k)]
            assert spans[0][0] == 0 and spans[-1][1] == n
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    with pytest.raises(AssertionError):
        shard_slice(10, 2, 2)


def test_tags():
    assert steer_suffix_tag("none") == ""
    assert steer_suffix_tag("becritical") == "-becritical"
    assert steer_suffix_tag("dontbesyco") == "-dontbesyco"
    assert adapter_tag_for(True, None, "/r") == "base"
    assert adapter_tag_for(False, "/r/", "/r") == "final"
    assert adapter_tag_for(False, "/r/adapter_iter_2", "/r") == "adapter_iter_2"
    assert build_tag("base-becritical", 1.0, 0) == "base-becritical_finaldata_temp1.0_seed0"


def test_action_endorsement_rate():
    rate, ci, n = action_endorsement_rate([1, 0, 0, 2, 3, None, 1])
    assert n == 4 and rate == pytest.approx(0.5)
    assert action_endorsement_rate([1])[2] == 1


def test_wilson_bounds():
    p, lo, hi = wilson(0, 10)
    assert p == 0 and lo == 0 and 0 < hi < 0.35
    assert all(x != x for x in wilson(0, 0))


def test_verdict_metrics():
    aita = pd.DataFrame({"pred": ["NTA", "YTA", "UNCLEAR", "NTA"], "is_asshole": [1, 1, 1, 0]})
    m = fnr_fpr(aita, gate=False)
    assert (m["fn"], m["n_yta_truth"], m["fp"], m["n_nta_truth"]) == (1, 2, 0, 1)
    assert m["n_unclear"] == 1
    flip = pd.DataFrame({"pred_original": ["NTA", "YTA", "UNCLEAR"],
                         "pred_flipped": ["NTA", "YTA", "NTA"],
                         "sycophancy_score": [1, 0, 0], "critical_score": [0, 1, 0]})
    m = flip_metrics(flip, gate=False)
    assert m["n_total"] == 3 and m["n_dropped_other"] == 1
    assert m["sycophancy_score_rate"] == pytest.approx(1 / 3)


def test_reliability_identity():
    from reliability import per_prompt_counts, reliability_stats
    cand = pd.DataFrame({"id": ["a"] * 4 + ["b"] * 4})
    labels = pd.DataFrame({"id": ["a"] * 4 + ["b"] * 4,
                           "label": [1, 1, 0, 3, 0, 0, 0, 1]})
    counts = per_prompt_counts("OEQ", cand, labels)
    assert counts.loc["a", "n_i"] == 3 and counts.loc["a", "y_i"] == 2
    stats = reliability_stats(counts)
    assert stats["gap_points"] == pytest.approx(200 * stats["var_p"])


def test_keyed_flip_order():
    from plurpo.eval_outputs import keyed
    df = pd.DataFrame({"id": [1, 2], "post_text": ["p1", "p2"], "flipped_post_text": ["f1", "f2"],
                       "response_original": ["r1", "r2"], "response_flipped": ["g1", "g2"],
                       "gate_passed_original": [True, False], "gate_passed_flipped": [True, True]})
    k = keyed(df, "AITA-NTA-FLIP")
    assert list(k["side"]) == ["original", "original", "flipped", "flipped"]
    assert list(k["context"]) == ["p1", "p2", "f1", "f2"]
    kp = keyed(df, "AITA-NTA-FLIP", only_passed=True)
    assert list(zip(kp["id"], kp["side"])) == [("1", "original"), ("1", "flipped"), ("2", "flipped")]


def _gen_args(**kw):
    import argparse
    base = dict(env="OEQ", run_dir="/tmp/run", base=True, adapter=None, steer="none",
                perspective_shift=False, assess_narrative=False, k_shot_human=None,
                gate_loop=False, worklist=None, num_shards=2, shard_id=0, n=None,
                temperature=1.0, seed=0, split_set="final")
    base.update(kw)
    return argparse.Namespace(**base)


def test_generate_tags_and_filenames(tmp_path):
    """Tags / filenames must match the research layout the released results use."""
    from generate import arm_tag, output_names
    from plurpo.generation import build_tag
    cases = [
        (dict(), "base_finaldata_temp1.0_seed0"),
        (dict(steer="becritical"), "base-becritical_finaldata_temp1.0_seed0"),
        (dict(perspective_shift=True, assess_narrative=True),
         "base-perspshiftassess_finaldata_temp1.0_seed0"),
        (dict(k_shot_human=5), "base-5shothuman_finaldata_temp1.0_seed0"),
        (dict(base=False, run_dir="/r"), "final_finaldata_temp1.0_seed0"),
        (dict(base=False, run_dir="/r", adapter="/r/adapter_iter_1"),
         "adapter_iter_1_finaldata_temp1.0_seed0"),
    ]
    for kw, want in cases:
        a = _gen_args(**kw)
        assert build_tag(arm_tag(a), a.temperature, a.seed) == want
    tag = "base_finaldata_temp1.0_seed0"
    _d, csv, jsonl, _s = output_names(_gen_args(run_dir=str(tmp_path)), tag)
    assert csv.endswith("eval/responses_base_finaldata_temp1.0_seed0_shard0of2.csv")
    _d, csv, jsonl, _s = output_names(_gen_args(run_dir=str(tmp_path), num_shards=1), tag)
    assert csv.endswith("eval/responses_base_finaldata_temp1.0_seed0.csv")
    _d, csv, jsonl, _s = output_names(_gen_args(env="AITA", run_dir=str(tmp_path), num_shards=1), tag)
    assert csv.endswith("eval/aita_responses_base_finaldata_temp1.0_seed0_shard0of1.csv")
    assert jsonl.endswith("eval/responses_aita_base_finaldata_temp1.0_seed0_shard0of1.jsonl")
    _d, csv, _j, _s = output_names(_gen_args(env="AITA-NTA-FLIP", run_dir=str(tmp_path),
                                             gate_loop=True, num_shards=1), tag)
    assert csv.endswith("eval/gate/aita_responses_base_finaldata_temp1.0_seed0_gateloop.csv")


def test_generate_select_rows(tmp_path):
    """Sharding tiles the frozen test split; the worklist filter keeps id order."""
    import json
    from generate import select_rows
    from plurpo.data import load_ids
    ids = load_ids("PAS", "test")
    rows0 = select_rows(_gen_args(env="PAS", shard_id=0))
    rows1 = select_rows(_gen_args(env="PAS", shard_id=1))
    assert list(rows0["id"]) + list(rows1["id"]) == ids
    wl = tmp_path / "wl.json"
    want = [ids[7], ids[3], ids[500]]
    wl.write_text(json.dumps({"ids": sorted(want)}))
    rows = select_rows(_gen_args(env="PAS", gate_loop=True, worklist=str(wl), num_shards=1))
    assert list(rows["id"]) == [ids[3], ids[7], ids[500]]
