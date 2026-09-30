"""The paper's main-text numbers, recomputed from results/, must equal the
published values. Values are copied from the paper (figure values from the
drawn labels / the data files the published figures were rendered from).

Known discrepancy (asserted explicitly, not hidden): Table 1's PlurPO OEQ cell
reads 0.282 [0.254, 0.310], which came from the first-draw PlurPO responses
(282/1000; not included in this release); every other cell of the table, and
Figure 2, use the rejection-sampled frames, under which PlurPO OEQ is
0.284 [0.256, 0.312] (284/1000).
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

import tables_main as T  # noqa: E402
from common import ENDORSE_ARMS_8B, load_arm  # noqa: E402
from plurpo import metrics as M  # noqa: E402

# Figure 2 (fig:oeq_pas_endorsement): arm -> (rate [Wilson CI], n01)
FIG2 = {
    "OEQ": {"human": ("0.496 [0.446, 0.546]", 383), "plurpo": ("0.486 [0.428, 0.544]", 284),
            "dpo_neutral": ("0.705 [0.596, 0.795]", 78),
            "becritical": ("0.601 [0.539, 0.659]", 253),
            "perspshift": ("0.602 [0.503, 0.693]", 98), "explicit": ("0.753 [0.680, 0.815]", 154),
            "sharma": ("0.819 [0.764, 0.863]", 232), "dontbesyco": ("0.830 [0.776, 0.873]", 229),
            "base": ("0.826 [0.747, 0.885]", 115)},
    "PAS": {"plurpo": ("0.028 [0.019, 0.041]", 855), "dpo_neutral": ("0.000 [0.000, 0.390]", 6),
            "perspshift": ("0.041 [0.017, 0.092]", 123),
            "becritical": ("0.068 [0.049, 0.094]", 485), "explicit": ("0.211 [0.165, 0.265]", 256),
            "sharma": ("0.319 [0.267, 0.376]", 276), "dontbesyco": ("0.448 [0.393, 0.504]", 308),
            "base": ("0.526 [0.447, 0.603]", 154)},
}
# Figure 3 (fig:verdict-errors): arm -> (x, xlo, xhi, y, ylo, yhi)
#   AITA x = FPR, y = FNR; AITA-Flipped x = both-YTA, y = both-NTA
FIG3 = {
    "AITA": {"base": (0.042, 0.028, 0.063, 0.898, 0.868, 0.922),
             "explicit": (0.018, 0.010, 0.034, 0.932, 0.906, 0.951),
             "sharma": (0.098, 0.075, 0.127, 0.834, 0.798, 0.864),
             "dontbesyco": (0.065, 0.046, 0.090, 0.852, 0.818, 0.880),
             "becritical": (0.100, 0.077, 0.130, 0.801, 0.764, 0.834),
             "perspshift": (0.066, 0.048, 0.092, 0.844, 0.809, 0.873),
             "plurpo": (0.624, 0.581, 0.666, 0.219, 0.184, 0.257)},
    "AITA-NTA-FLIP": {"base": (0.007, 0.003, 0.014, 0.859, 0.836, 0.879),
                      "explicit": (0.000, 0.000, 0.004, 0.870, 0.848, 0.889),
                      "sharma": (0.014, 0.008, 0.024, 0.776, 0.749, 0.801),
                      "dontbesyco": (0.012, 0.007, 0.021, 0.761, 0.734, 0.786),
                      "becritical": (0.032, 0.023, 0.045, 0.690, 0.660, 0.718),
                      "perspshift": (0.011, 0.006, 0.020, 0.748, 0.720, 0.775),
                      "plurpo": (0.200, 0.176, 0.226, 0.244, 0.218, 0.272)},
}
# Table 1 (tab:explicit-frac): arm -> (OEQ, PAS), Wald CIs
TAB1 = {"base": ("0.115 [0.095, 0.135]", "0.157 [0.134, 0.180]"),
        "becritical": ("0.253 [0.226, 0.280]", "0.511 [0.479, 0.543]"),
        "explicit": ("0.154 [0.132, 0.176]", "0.256 [0.229, 0.283]"),
        "sharma": ("0.232 [0.206, 0.258]", "0.276 [0.249, 0.304]"),
        "dontbesyco": ("0.229 [0.203, 0.255]", "0.311 [0.283, 0.340]"),
        "perspshift": ("0.104 [0.084, 0.123]", "0.173 [0.145, 0.201]"),
        "dpo_neutral": ("0.078 [0.061, 0.095]", "0.006 [0.001, 0.011]"),
        "plurpo": ("0.282 [0.254, 0.310]", "0.855 [0.833, 0.877]")}
# Figure 4 top (fig:pas-elephant): arm -> (% accepts framing, % validates emotions)
FIG4 = {"plurpo": (26.5, 46.5), "dpo_neutral_nonmixed": (76.5, 99.6),
        "becritical": (20.8, 60.2), "explicit": (48.1, 81.4), "perspshift": (34.2, 34.6),
        "sharma": (65.6, 78.9), "dontbesyco": (70.1, 75.5), "base": (82.9, 89.4)}


@pytest.mark.parametrize("env", ["OEQ", "PAS"])
def test_figure2_endorsement(env):
    rows = {r["arm"]: r for r in T.endorsement_rows("qwen3-8b", env, ENDORSE_ARMS_8B[env])}
    assert set(rows) == set(FIG2[env])
    for arm, (ci, n01) in FIG2[env].items():
        r = rows[arm]
        assert M.fmt_ci(r["rate"], r["lo"], r["hi"]) == ci, (env, arm)
        assert r["n01"] == n01, (env, arm)


@pytest.mark.parametrize("env", ["AITA", "AITA-NTA-FLIP"])
def test_figure3_verdict_errors(env):
    rows = {r["arm"]: r for r in T.verdict_rows("qwen3-8b", env)}
    for arm, want in FIG3[env].items():
        got = tuple(round(v, 3) for v in tuple(rows[arm]["x"]) + tuple(rows[arm]["y"]))
        assert got == pytest.approx(want, abs=1e-9), (env, arm)


def test_table1_explicit_stance():
    for arm, (oeq, pas) in TAB1.items():
        got_pas = M.fmt_ci(*M.explicit_stance_fraction(load_arm("qwen3-8b", "PAS", arm))[:3])
        assert got_pas == pas, arm
        got_oeq = M.fmt_ci(*M.explicit_stance_fraction(load_arm("qwen3-8b", "OEQ", arm))[:3])
        if arm == "plurpo":
            continue          # known discrepancy, see test below
        assert got_oeq == oeq, arm


def test_table1_plurpo_oeq_known_discrepancy():
    gated = M.fmt_ci(*M.explicit_stance_fraction(load_arm("qwen3-8b", "OEQ", "plurpo"))[:3])
    assert gated == "0.284 [0.256, 0.312]"


def test_figure4_pas_elephant():
    rows = {r["arm"]: r for r in T.elephant_rows("PAS")}
    assert set(rows) == set(FIG4)
    for arm, (fr, va) in FIG4.items():
        assert round(100 * rows[arm]["framing"][0], 1) == fr, arm
        assert round(100 * rows[arm]["validation"][0], 1) == va, arm
