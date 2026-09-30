"""GEPA metrics reproduce the research metrics exactly: score and the
natural-language feedback (the reflection LM's input) for every
(human label, model label) combination. Expected values were produced by the
ORIGINAL dspy_baselines metrics with the same stubbed judge; no API call."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "baselines", "gepa"))


def test_gepa_metrics_match_research():
    import dspy
    import gepa_lib
    gepa_lib.judge_endorse = lambda client, tracker, post, resp, arm: int(resp)
    with open(os.path.join(HERE, "fixtures", "gepa_metric_expected.json")) as f:
        cases = json.load(f)
    for human, model, score, feedback in cases:
        pred = dspy.Prediction(response=str(model))
        if human is None:
            got = gepa_lib.make_oeq_resolved_metric(None, None, "x")(
                dspy.Example(post_text="p"), pred)
        else:
            got = gepa_lib.make_oeq_metric(None, None)(
                dspy.Example(post_text="p", human_label=human), pred)
        assert got.score == score and got.feedback == feedback


def test_rate_match_adapter_scoring():
    import run_rate_match as R

    class Batch:
        def __init__(self, scores):
            self.scores = scores

    adapter = R.RateMatchAdapter.__new__(R.RateMatchAdapter)
    adapter.target_rate = 0.5
    orig = R.DspyAdapter.evaluate
    R.DspyAdapter.evaluate = lambda self, b, c, capture_traces=False: Batch(list(b))
    out = adapter.evaluate([1.0, 0.0, 1.0, 3.0], None)
    assert out.scores == [1 - abs(2 / 3 - 0.5)] * 4
    out = adapter.evaluate([2.0, 3.0], None)
    assert out.scores == [0.0, 0.0]
    R.DspyAdapter.evaluate = orig
