"""Per-model run configurations (configs/<name>.json).

A config holds everything that differs between model families (model id, LoRA
target modules, veto prompt, candidate mix, number of epochs, ...). A value may
be a per-environment dict keyed by env name (e.g. `perspshift_assess`), which
`per_env` resolves.
"""

import json
import os

from plurpo.data import ENVS
from plurpo.paths import REPO_ROOT

CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def load_config(name):
    """Load configs/<name>.json (or an explicit .json path)."""
    path = name if name.endswith(".json") else os.path.join(CONFIG_DIR, f"{name}.json")
    assert os.path.exists(path), f"missing config {path}"
    with open(path) as f:
        return json.load(f)


def per_env(value, env):
    """Resolve a possibly per-env config value for `env`."""
    if isinstance(value, dict):
        assert set(value) == set(ENVS), f"per-env value must cover {ENVS}: {value}"
        return value[env]
    return value


def lora_targets(value):
    """"all-linear" stays a string; "a,b,c" or a list becomes a list."""
    if isinstance(value, list):
        return value
    if "," in value:
        return [m.strip() for m in value.split(",") if m.strip()]
    return value
