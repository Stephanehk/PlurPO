"""Shared loaders, method names and orderings for the analysis scripts.

Every table and figure reads ONLY files under `results/` (see
results/README.md). Per-arm frames live at results/<model>/<ENV>/<arm>.csv:
held-out responses AFTER rejection sampling (the paper's main protocol).

Arm keys: base, explicit (Cheng et al. 2026 prompt), sharma (Sharma et al.
2024 prompt), dontbesyco, becritical, perspshift, dpo_neutral (the paper's
DPO-Neutral), dpo_neutral_nonmixed, plurpo, plurpo_stage1 (PlurPO before the
role-confusion stage), human (see arms.csv).
"""

import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plurpo.paths import RESULTS_DIR  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "_reproduced")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
ENVS = ("OEQ", "PAS", "AITA", "AITA-NTA-FLIP")

_READ = dict(keep_default_na=False, na_values=[""], dtype={"id": str})


def load_arm(model, env, arm):
    """Rejection-sampled frame for one (model, env, arm)."""
    path = os.path.join(RESULTS_DIR, model, env, f"{arm}.csv")
    assert os.path.exists(path), f"missing result frame {path}"
    return pd.read_csv(path, **_READ)


def ensure_out():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


# ------------------------------------------------------------------ naming
OURS = "PlurPO"
HUMAN = "Human baseline"


def method_labels(model_name):
    """arm key -> display name, e.g. model_name="Qwen3-8B"."""
    return {
        "human": HUMAN,
        "plurpo": OURS,
        "dpo_neutral": "DPO-Neutral",
        "dpo_neutral_nonmixed": "DPO-Neutral",
        "perspshift": "Perspective-shift",
        "becritical": f'{model_name} + "Be critical"',
        "dontbesyco": f'{model_name} + "Don\'t-be-sycophantic"',
        "explicit": f"{model_name} + Cheng et al. (2026) prompt",
        "sharma": f"{model_name} + Sharma et al. (2024) prompt",
        "base": model_name,
    }


# Canonical order, top -> bottom (legend order for scatter plots).
CANON = ["human", "plurpo", "dpo_neutral", "dpo_neutral_nonmixed", "perspshift",
         "becritical", "dontbesyco", "explicit", "sharma", "base"]


def ordered(arms):
    """Sort arm keys by CANON; asserts every key is known."""
    for a in arms:
        assert a in CANON, f"no canonical position for arm {a!r}"
    return sorted(arms, key=CANON.index)


# Arms shown in each Qwen3-8B main figure (the paper's rows).
ENDORSE_ARMS_8B = {
    "OEQ": ordered(["human", "plurpo", "dpo_neutral", "perspshift", "becritical",
                    "dontbesyco", "explicit", "sharma", "base"]),
    "PAS": ordered(["plurpo", "dpo_neutral", "perspshift", "becritical",
                    "dontbesyco", "explicit", "sharma", "base"]),
}
VERDICT_ARMS = ordered(["plurpo", "perspshift", "becritical", "dontbesyco",
                        "explicit", "sharma", "base"])
# The paper's ELEPHANT-judge figures used the non-mixed Neutral DPO run for their
# "DPO-Neutral" row (the endorsement figures use the mixed run). Reproduced as
# published; see results/README.md.
ELEPHANT_ARMS = {
    "OEQ": ordered(["human", "plurpo", "dpo_neutral_nonmixed", "perspshift", "becritical",
                    "dontbesyco", "explicit", "sharma", "base"]),
    "PAS": ordered(["plurpo", "dpo_neutral_nonmixed", "perspshift", "becritical",
                    "dontbesyco", "explicit", "sharma", "base"]),
    "AITA": VERDICT_ARMS,
    "AITA-NTA-FLIP": VERDICT_ARMS,
}


# The published figures plotted rates and CI bounds parsed from 3-decimal table
# strings. By default the figure scripts round the same way so the PNGs are
# pixel-identical to the paper; set PLURPO_EXACT_PLOTS=1 to plot full precision
# (differences are sub-pixel: at most 0.0005 on a 0-1 axis).
PAPER_ROUNDING = os.environ.get("PLURPO_EXACT_PLOTS") != "1"


def r3(x):
    """Round a plotted value like the published figures (see PAPER_ROUNDING)."""
    if x is None or not PAPER_ROUNDING:
        return x
    return round(float(x), 3)


def write_table(df, name):
    """Write a reproduced table as CSV and print it."""
    ensure_out()
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))
    return path
