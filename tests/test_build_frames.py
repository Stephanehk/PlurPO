"""scripts/analysis/build_frames.py must rebuild the shipped result frames from
evaluation-pipeline outputs (cell-identical).

Runs offline against the ORIGINAL research run dirs / scored files, whose
layout the new pipeline reproduces. The one layout difference -- the research
code pooled all gate-rescued ELEPHANT labels into one file per (env, metric),
while judge_elephant.py writes one file per arm with id/side -- is bridged by
converting the pooled file into per-arm files first. Skipped where the research
artifacts are not mounted.
"""

import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))
sys.path.insert(0, REPO)

# Optional: PLURPO_RESEARCH_ROOT holding the original runs/ and sycoharm_runs/.
RESEARCH_ROOT = os.environ.get("PLURPO_RESEARCH_ROOT", "")
RUNS = os.path.join(RESEARCH_ROOT, "runs")
SYCO = os.path.join(RESEARCH_ROOT, "sycoharm_runs")
JD = f"{SYCO}/judges_finaldata"
JDIR = {"OEQ": "OEQ", "PAS": "PAS", "AITA": "AITA", "AITA-NTA-FLIP": "FLIP"}

pytestmark = pytest.mark.skipif(not RESEARCH_ROOT or not os.path.isdir(RUNS) or not os.path.isdir(SYCO),
                                reason="research artifacts not available")

READ = dict(keep_default_na=False, na_values=[""], dtype={"id": str})


def elephant_dir(tmp_path, env, jslug, rslug):
    """Per-arm elephant files in the judge_elephant.py layout: the attempt-1
    files as-is, plus <metric>_<rslug>_gateloop.csv converted from the
    research pooled rescued file (rows of this arm that FAILED at attempt 1)."""
    sys.path.insert(0, os.path.join(REPO, "results", "_provenance"))
    from elephant_join import rescued_keys
    d = tmp_path / f"elephant_{env}_{jslug}"
    d.mkdir()
    for metric in ("framing", "validation"):
        os.symlink(f"{JD}/{JDIR[env]}/{metric}_{jslug}.csv", d / f"{metric}_{jslug}.csv")
        k = rescued_keys(env, metric)
        k = k[(k["arm"] == rslug) & k["is_rescued"]]
        pd.DataFrame({"id": k["id"], "side": k["side"], "response": k["response"],
                      f"{metric}_score_response": k[f"{metric}_response"]}).to_csv(
            d / f"{metric}_{rslug}_gateloop.csv", index=False)
    return str(d)


def build_and_compare(tmp_path, argv, shipped):
    import build_frames
    out = str(tmp_path / "frame.csv")
    build_frames.main(argv + ["--out", out])
    got = pd.read_csv(out, **READ)
    want = pd.read_csv(os.path.join(REPO, "results", shipped), **READ)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


CASES = [  # (env, arm, run, tag, endorse name, gateloop name, v1 name, jslug, rslug)
    ("OEQ", "base", "baseline_final_OEQ", "base", "oeq_base_finaldata", None,
     "oeq_qwen38bbase", "base", "base"),
    ("OEQ", "plurpo", "dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5", "final",
     "oeq_voice2_finaldata", "oeq_voice2_gateloop", "oeq_ourmodel31", "voice2", "voice2"),
    ("AITA-NTA-FLIP", "base", "baseline_final_AITA-NTA-FLIP", "base", None, None, None,
     "base", "base"),
    ("AITA-NTA-FLIP", "plurpo", "dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5", "final",
     None, None, None, "voice2", "voice2"),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_rejection_sampled_frames(tmp_path, case):
    env, arm, run, tag, ename, gname, v1, jslug, rslug = case
    argv = ["--model", "qwen3-8b", "--env", env, "--arm", arm, "--run-dir", f"{RUNS}/{run}",
            "--tag", tag, "--elephant-dir", elephant_dir(tmp_path, env, jslug, rslug),
            "--elephant-name", jslug, "--elephant-gateloop-name", f"{rslug}_gateloop"]
    if ename:
        argv += ["--scores-root", SYCO, "--endorse-name", ename]
    if gname:
        argv += ["--endorse-gateloop-name", gname]
    if v1:
        argv += ["--v1-name", v1]
    build_and_compare(tmp_path, argv, f"qwen3-8b/{env}/{arm}.csv")


def test_human_frame(tmp_path):
    d = tmp_path / "eh"
    d.mkdir()
    for metric in ("framing", "validation"):
        os.symlink(f"{JD}/OEQ/{metric}_human.csv", d / f"{metric}_human.csv")
    build_and_compare(tmp_path, ["--model", "qwen3-8b", "--env", "OEQ", "--arm", "human",
                                 "--scores-root", SYCO, "--endorse-name", "oeq_human_finaldata",
                                 "--v1-name", "oeq_humanbaseline", "--elephant-dir", str(d),
                                 "--elephant-name", "human"], "qwen3-8b/OEQ/human.csv")
