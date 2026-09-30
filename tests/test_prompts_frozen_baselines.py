"""The GEPA baselines' seed instruction and field descriptions are
byte-identical to the research code (sha256 pinned from the original
dspy_baselines/program.py). The metric feedback strings, which GEPA's
reflection LM reads, are pinned by tests/test_baselines_gepa.py."""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "baselines", "gepa"))


def test_gepa_program_text_unchanged():
    import gepa_lib
    s = gepa_lib.RespondToPost
    vals = {"RespondToPost.instructions": s.instructions,
            "RespondToPost.post_text.desc": s.input_fields["post_text"].json_schema_extra["desc"],
            "RespondToPost.response.desc": s.output_fields["response"].json_schema_extra["desc"]}
    with open(os.path.join(HERE, "fixtures", "prompt_sha256_baselines.json")) as f:
        expected = json.load(f)
    assert set(expected) == set(vals)
    for key, digest in expected.items():
        assert hashlib.sha256(vals[key].encode()).hexdigest() == digest, f"{key} was modified"
