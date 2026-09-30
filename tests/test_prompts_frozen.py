"""Every training-time prompt is byte-identical to the one that produced the
paper's results.

`fixtures/prompt_sha256_training.json` maps "<module>.<CONSTANT>" to the
sha256 of the ORIGINAL research-code constant (computed once against that code
when this release was assembled). Any edit to a prompt fails this test.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from plurpo import perspective_shift
from plurpo.prompts import baselines, plurpo, role_confusion

MODULES = {"plurpo": plurpo, "baselines": baselines, "role_confusion": role_confusion,
           "perspective_shift": perspective_shift}


def test_training_prompts_unchanged():
    with open(os.path.join(HERE, "fixtures", "prompt_sha256_training.json")) as f:
        expected = json.load(f)
    for key, digest in expected.items():
        module, name = key.split(".")
        value = getattr(MODULES[module], name)
        assert hashlib.sha256(value.encode()).hexdigest() == digest, f"{key} was modified"


if __name__ == "__main__":
    test_training_prompts_unchanged()
    print("ok")
