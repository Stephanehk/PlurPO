"""Unit tests for plurpo/metrics.py (pure functions, synthetic frames)."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plurpo import metrics as M  # noqa: E402


def test_wilson_known_value():
    p, lo, hi = M.wilson(95, 115)
    assert M.fmt_ci(p, lo, hi) == "0.826 [0.747, 0.885]"


def test_wilson_zero_n_is_nan():
    p, lo, hi = M.wilson(0, 0)
    assert p != p and lo != lo and hi != hi


def test_wald_clips_to_unit_interval():
    p, lo, hi = M.wald(0, 10)
    assert (p, lo, hi) == (0.0, 0.0, 0.0)
    assert M.fmt_ci(*M.wald(282, 1000)) == "0.282 [0.254, 0.310]"


def _endorse_frame(labels, gate):
    return pd.DataFrame({"endorse_response": pd.array(labels, dtype="Int64"),
                         "gate_passed": gate, "n_tried": [1] * len(labels)})


def test_endorsement_drops_ungated_and_uses_explicit_denominator():
    f = _endorse_frame([1, 1, 0, 2, 3, None], [1, 1, 1, 1, 1, 0])
    c = M.endorsement_counts(f)
    assert (c["n_scored"], c["n01"], c["n1"]) == (5, 3, 2)
    p, _lo, _hi, n01 = M.endorsement_rate(f)
    assert n01 == 3 and p == pytest.approx(2 / 3)
    p, _lo, _hi, n = M.explicit_stance_fraction(f)
    assert n == 5 and p == pytest.approx(3 / 5)


def test_endorsement_asserts_on_missing_gated_label():
    f = _endorse_frame([1, None], [1, 1])
    with pytest.raises(AssertionError):
        M.endorsement_counts(f)


def test_attempt1_frames_are_not_gated():
    f = pd.DataFrame({"endorse_response": [1, 0, 0]})
    assert M.endorsement_counts(f)["n_scored"] == 3


def test_fnr_fpr_excludes_unclear_and_dropped():
    f = pd.DataFrame({"is_asshole": [1, 1, 1, 0, 0, 0],
                      "pred": ["NTA", "YTA", "UNCLEAR", "YTA", "NTA", "NTA"],
                      "gate_passed": [1, 1, 1, 1, 1, 0]})
    m = M.fnr_fpr(f)
    assert (m["fn"], m["fp"], m["n_yta_truth"], m["n_nta_truth"], m["n_unclear"]) == (1, 1, 2, 2, 1)


def test_flip_keeps_unclear_pairs_in_denominator_and_drops_failed_sides():
    f = pd.DataFrame({"pred_original": ["NTA", "YTA", "UNCLEAR", "NTA"],
                      "pred_flipped": ["NTA", "YTA", "NTA", "NTA"],
                      "gate_passed_original": [1, 1, 1, 1],
                      "gate_passed_flipped": [1, 1, 1, 0]})
    assert M.flip_rates(f) == {"n_both_nta": 1, "n_both_yta": 1, "n_pairs": 3}
    assert M.n_filtered(f, "AITA-NTA-FLIP") == 1


def test_macro_f1_perfect_classifier():
    f = pd.DataFrame({"is_asshole": [1, 1, 0, 0], "pred": ["YTA", "YTA", "NTA", "NTA"]})
    assert M.macro_f1(f) == pytest.approx(1.0)


def test_sample_counts_and_mean_ci():
    f = pd.DataFrame({"gate_passed": [1, 1, 0], "n_tried": [1, 3, 5]})
    assert M.sample_counts(f, "OEQ") == [1, 3]
    mean, lo, hi = M.mean_and_ci([1, 1, 1])
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)


def test_mcnemar_exact():
    p, a, b = M.mcnemar_exact([1, 1, 0, 0], [1, 1, 0, 0])
    assert (p, a, b) == (1.0, 0, 0)
    p, a, b = M.mcnemar_exact([1] * 10 + [0] * 10, [0] * 10 + [0] * 10)
    assert (a, b) == (10, 0) and p == pytest.approx(2 * 0.5 ** 10)
