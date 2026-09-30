"""Where every shipped result frame came from (original research artifacts).

Provenance only: these paths point at the original research artifacts and
exist so `build_results.py` can be re-run against them.
Set PLURPO_RESEARCH_ROOT to a directory laid out as
    <research>/runs/                         per-run generation + gate outputs
    <research>/sycoharm_runs/                judge outputs (endorse_*, judges_finaldata/, ...)
Paths recorded in results/ use the literal prefix "<research>".
Nothing in `plurpo/` or `scripts/analysis/` imports this module.

ARMS rows: (model, env, arm, source kind, run subdir, gen tag, extra dict)
  source kind "inloop"  -- rejection sampling ran inside the generation loop;
                            artifacts: gate_<TAG>_a1.csv + *_gateloop.csv
  source kind "staged"  -- rejection sampling ran as separate rounds;
                            artifacts: gatefinal_<TAG>.csv + gatestats_<TAG>.json
  source kind "human"   -- OEQ human responses, never gated.
extra keys:
  v5       frozen attempt-1 gpt-5-mini v5 endorsement file (OEQ/PAS)
  v5g      v5 file for gate-rescued responses (OEQ/PAS, basename stem)
  jslug    ELEPHANT judge file slug (judges_finaldata/<ENV>/<metric>_<slug>.csv)
  rslug    arm label inside judges_finaldata/_gated_rescued/*
  v1       original-prompt (v1) endorsement file (already gated)
"""

import os

ROOT = os.environ.get("PLURPO_RESEARCH_ROOT", "")
RUNS = os.path.join(ROOT, "runs")
SYCO = os.path.join(ROOT, "sycoharm_runs")


def require_root():
    """Fail loudly unless PLURPO_RESEARCH_ROOT points at the research artifacts."""
    assert ROOT and os.path.isdir(RUNS) and os.path.isdir(SYCO), (
        "set PLURPO_RESEARCH_ROOT to the directory holding runs/ and sycoharm_runs/")


def anon(path):
    """Record a research path with the neutral '<research>' prefix."""
    assert ROOT, "PLURPO_RESEARCH_ROOT is not set"
    return path.replace(ROOT.rstrip("/"), "<research>", 1)
V5 = f"{SYCO}/endorse_gpt5mini_v5_temp0"
V5G = f"{V5}_gateloop"
V1 = f"{SYCO}/endorse_gpt5mini_v1_gated"
JD = f"{SYCO}/judges_finaldata"
TAGSUF = "_finaldata_temp1.0_seed0"

STEERS = [("base", "base"), ("explicit", "base-explicit"), ("sharma", "base-sharma"),
          ("dontbesyco", "base-dontbesyco"), ("becritical", "base-becritical"),
          ("perspshift", "base-perspshiftassess")]
# v5 frozen file slug and v1 file slug per steer arm (8B)
V5SLUG = {"base": "base", "explicit": "explicit", "sharma": "sharma",
          "dontbesyco": "dontbesyco", "becritical": "becritical",
          "perspshift": "perspshiftassess"}
JSLUG = dict(V5SLUG)
RSLUG = {"base": "base", "explicit": "explicit", "sharma": "sharma",
         "dontbesyco": "dontbesyco", "becritical": "becritical",
         "perspshift": "perspshift"}
V1SLUG = {"base": "qwen38bbase", "explicit": "chengetalprompt",
          "sharma": "sharmaetalprompt", "dontbesyco": "dontbesycophantic",
          "becritical": "becritical", "perspshift": "perspectiveshift",
          "dpo_neutral": "neutralmixeddpo", "plurpo": "ourmodel31",
          "human": "humanbaseline"}
# gate-rescued v5 files for the 8B OEQ/PAS arms (from scratch_gate_manifest.csv)
V5G_8B = {
    ("OEQ", "becritical"): "oeq_base_oeq_basebecritical",
    ("OEQ", "perspshift"): "oeq_base_oeq_baseperspshiftassess",
    ("PAS", "becritical"): "pas_base_pas_basebecritical",
    ("PAS", "dontbesyco"): "pas_base_pas_basedontbesyco",
    ("PAS", "explicit"): "pas_base_pas_baseexplicit",
    ("PAS", "perspshift"): "pas_base_pas_baseperspshiftassess",
    ("PAS", "sharma"): "pas_base_pas_basesharma",
    ("PAS", "base"): "pas_base_pas_base",
    ("OEQ", "plurpo_stage1"): "oeq_oeq_mixed_p6_perspshift_distill_gp_rpo0p5_final_final",
    ("PAS", "plurpo_stage1"): "pas_pas_mixed_p6_perspshift_distill_gp_rpo0p5_final_final",
    ("OEQ", "dpo_neutral"): "oeq_oeq_neutral_mixed_final_final",
    ("PAS", "dpo_neutral"): "pas_pas_neutral_mixed_final_final",
    ("PAS", "dpo_neutral_nonmixed"): "pas_pas_neutral_final_final",
    ("OEQ", "plurpo"): "oeq_voice2",
    ("PAS", "plurpo"): "pas_voice2",
}
BASELINE_RUN = {"OEQ": "baseline_final_OEQ", "PAS": "baseline_final_PAS",
                "AITA": "baseline_final_AITA",
                "AITA-NTA-FLIP": "baseline_final_AITA-NTA-FLIP"}
STAGE1_RUN = {
    "OEQ": "dpo_lora_oeq_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
    "PAS": "dpo_lora_pas_qwen3-8b_n1000_mixed_P6_perspshift_distill_gp_rpo0p5_final",
}
PLURPO_RUN = {"OEQ": "dpo_lora_oeq_qwen3-8b_P6_voice2_rpo0p5",
              "PAS": "dpo_lora_pas_qwen3-8b_P6_voice2_rpo0p5",
              "AITA": "dpo_lora_aita_qwen3-8b_P6_voice2R2_rpo0p5",
              "AITA-NTA-FLIP": "dpo_lora_flip_qwen3-8b_P6_voice2R2_rpo0p5"}
NEUTRAL_RUN = {"OEQ": "dpo_lora_oeq_qwen3-8b_n1000_neutral_mixed_final",
               "PAS": "dpo_lora_pas_qwen3-8b_n1000_neutral_mixed_final"}
NEUTRAL_NONMIXED_RUN = {"OEQ": "dpo_lora_oeq_qwen3-8b_n1000_neutral_final",
                        "PAS": "dpo_lora_pas_qwen3-8b_n1000_neutral_final"}


def _arms_8b():
    arms = []
    for env in ("OEQ", "PAS"):
        e = env.lower()
        for arm, tag in STEERS:
            arms.append(("qwen3-8b", env, arm, "inloop", BASELINE_RUN[env], tag, {
                "v5": f"scored_{e}_{V5SLUG[arm]}_finaldata.csv",
                "v5g": V5G_8B.get((env, arm)), "jslug": JSLUG[arm],
                "rslug": RSLUG[arm], "v1": f"scored_{e}_{V1SLUG[arm]}.csv"}))
        arms.append(("qwen3-8b", env, "dpo_neutral", "inloop", NEUTRAL_RUN[env], "final", {
            "v5": f"scored_{e}_neutralmixed_finaldata.csv",
            "v5g": V5G_8B.get((env, "dpo_neutral")), "jslug": "neutralmixed",
            "rslug": "neutralmixed", "v1": f"scored_{e}_neutralmixeddpo.csv"}))
        arms.append(("qwen3-8b", env, "dpo_neutral_nonmixed", "inloop",
                     NEUTRAL_NONMIXED_RUN[env], "final", {
                         "v5": f"scored_{e}_neutral_finaldata.csv",
                         "v5g": V5G_8B.get((env, "dpo_neutral_nonmixed")),
                         "jslug": "neutral", "rslug": "neutral", "v1": None}))
        arms.append(("qwen3-8b", env, "plurpo_stage1", "inloop", STAGE1_RUN[env], "final", {
            "v5": f"scored_{e}_P6perspshift_a05_finaldata.csv",
            "v5g": V5G_8B.get((env, "plurpo_stage1")), "jslug": None,
            "rslug": None, "v1": None}))
        arms.append(("qwen3-8b", env, "plurpo", "inloop", PLURPO_RUN[env], "final", {
            "v5": f"scored_{e}_voice2_finaldata.csv", "v5g": V5G_8B[(env, "plurpo")],
            "jslug": "voice2", "rslug": "voice2", "v1": f"scored_{e}_ourmodel31.csv"}))
    arms.append(("qwen3-8b", "OEQ", "human", "human", None, None, {
        "v5": "scored_oeq_human_finaldata.csv", "jslug": "human",
        "v1": "scored_oeq_humanbaseline.csv"}))
    for env in ("AITA", "AITA-NTA-FLIP"):
        for arm, tag in STEERS:
            arms.append(("qwen3-8b", env, arm, "inloop", BASELINE_RUN[env], tag,
                         {"jslug": JSLUG[arm], "rslug": RSLUG[arm]}))
        arms.append(("qwen3-8b", env, "plurpo", "staged", PLURPO_RUN[env], "final",
                     {"jslug": "voice2", "rslug": "voice2"}))
    return arms


ARMS = _arms_8b()
