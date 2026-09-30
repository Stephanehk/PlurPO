"""Tables behind the paper's main-text results (and their direct appendix
companions), computed from results/ only.

  endorsement_qwen3-8b_{OEQ,PAS}              Fig. 2 (+ App. full panels):
      action endorsement rate [Wilson 95% CI], n01, n_scored, n_filtered,
      mean samples to pass rejection sampling [normal 95% CI], neutral rate.
  explicit_stance                             Table 1 (Wald 95% CI)
  neutral_rate_{OEQ,PAS}                      App. Neutral-DPO table
  verdict_qwen3-8b_{AITA,AITA-NTA-FLIP}       Fig. 3
  aita_macro_f1                               App. AITA macro-F1 figure
  elephant_{env}                              Fig. 4 (PAS) + App. ELEPHANT figure
  endorsement_v1judge_{OEQ,PAS}               App. original-judge figure (Wald CIs,
                                              as in the original table)

Run: python scripts/analysis/tables_main.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_arm, method_labels, write_table, ENDORSE_ARMS_8B,  # noqa: E402
                    VERDICT_ARMS, ELEPHANT_ARMS)
from plurpo import metrics as M  # noqa: E402

MODEL_NAME = {"qwen3-8b": "Qwen3-8B"}


def _arm_model(model, arm):
    """The OEQ human reference is shared by every model's table."""
    return "qwen3-8b" if arm == "human" else model


def endorsement_rows(model, env, arms):
    """One dict per arm for the endorsement-vs-samples figures."""
    labels = method_labels(MODEL_NAME[model])
    rows = []
    for arm in arms:
        f = load_arm(_arm_model(model, arm), env, arm)
        c = M.endorsement_counts(f)
        p, lo, hi = M.wilson(c["n1"], c["n01"])
        np_, nlo, nhi = M.wilson(c["n3"], c["n_scored"])
        row = {"arm": arm, "method": labels[arm], "rate": p, "lo": lo, "hi": hi,
               "n1": c["n1"], "n01": c["n01"], "n_scored": c["n_scored"],
               "neutral": np_, "neutral_lo": nlo, "neutral_hi": nhi,
               "n_filtered": None, "mean_samples": None, "samples_lo": None,
               "samples_hi": None}
        if arm != "human":          # the human reference was never gated
            ms, slo, shi = M.mean_and_ci(M.sample_counts(f, env))
            row.update(n_filtered=M.n_filtered(f, env), mean_samples=ms,
                       samples_lo=slo, samples_hi=shi)
        rows.append(row)
    return rows


def endorsement_table(model, env, arms):
    rows = endorsement_rows(model, env, arms)
    out = pd.DataFrame([{
        "method": r["method"],
        "endorsement rate [95% CI]": M.fmt_ci(r["rate"], r["lo"], r["hi"]),
        "n_endorse": r["n01"], "n_scored": r["n_scored"],
        "neutral rate [95% CI]": M.fmt_ci(r["neutral"], r["neutral_lo"], r["neutral_hi"]),
        "n_filtered": "n/a" if r["n_filtered"] is None else r["n_filtered"],
        "mean_samples": "" if r["mean_samples"] is None else f"{r['mean_samples']:.3f}",
    } for r in rows])
    return write_table(out, f"endorsement_{model}_{env}")


def explicit_stance_table():
    """Table 1: share of responses explicitly endorsing or challenging."""
    labels = method_labels("Qwen3-8B")
    arms = [a for a in ENDORSE_ARMS_8B["PAS"]]
    order = ["base", "becritical", "explicit", "sharma", "dontbesyco", "perspshift",
             "dpo_neutral", "plurpo"]          # the table's row order
    assert sorted(order) == sorted(arms)
    rows = []
    for arm in order:
        rec = {"method": labels[arm]}
        for env in ("OEQ", "PAS"):
            p, lo, hi, n = M.explicit_stance_fraction(load_arm("qwen3-8b", env, arm))
            rec[env] = M.fmt_ci(p, lo, hi)
        rows.append(rec)
    return write_table(pd.DataFrame(rows), "explicit_stance")


def neutral_table(env):
    rows = endorsement_rows("qwen3-8b", env, ENDORSE_ARMS_8B[env])
    out = pd.DataFrame([{"method": r["method"], "neutral rate [95% CI]":
                         M.fmt_ci(r["neutral"], r["neutral_lo"], r["neutral_hi"])}
                        for r in rows])
    return write_table(out, f"neutral_rate_{env}")


def verdict_rows(model, env):
    labels = method_labels(MODEL_NAME[model])
    rows = []
    for arm in VERDICT_ARMS:
        f = load_arm(model, env, arm)
        rec = {"arm": arm, "method": labels[arm], "n_filtered": M.n_filtered(f, env)}
        ms, slo, shi = M.mean_and_ci(M.sample_counts(f, env))
        rec.update(mean_samples=ms, samples_lo=slo, samples_hi=shi)
        if env == "AITA":
            m = M.fnr_fpr(f)
            rec["x"] = M.wilson(m["fp"], m["n_nta_truth"])      # FPR
            rec["y"] = M.wilson(m["fn"], m["n_yta_truth"])      # FNR
            rec.update(n_yta=m["n_yta_truth"], n_nta=m["n_nta_truth"])
        else:
            m = M.flip_rates(f)
            rec["x"] = M.wilson(m["n_both_yta"], m["n_pairs"])  # both-YTA
            rec["y"] = M.wilson(m["n_both_nta"], m["n_pairs"])  # both-NTA
            rec["n_pairs"] = m["n_pairs"]
        rows.append(rec)
    return rows


def verdict_table(model, env):
    rows = verdict_rows(model, env)
    xn, yn = (("FPR", "FNR (sycophancy)") if env == "AITA" else
              ("both-YTA", "both-NTA (sycophancy)"))
    out = pd.DataFrame([{
        "method": r["method"], f"{yn} [95% CI]": M.fmt_ci(*r["y"]),
        f"{xn} [95% CI]": M.fmt_ci(*r["x"]), "n_filtered": r["n_filtered"],
        "mean_samples": f"{r['mean_samples']:.3f}",
        **({"n_YTA_truth": r["n_yta"], "n_NTA_truth": r["n_nta"]} if env == "AITA"
           else {"n_pairs": r["n_pairs"]}),
    } for r in rows])
    return write_table(out, f"verdict_{model}_{env}")


def macro_f1_rows():
    labels = method_labels("Qwen3-8B")
    rows = []
    for arm in VERDICT_ARMS:
        f = load_arm("qwen3-8b", "AITA", arm)
        lo, hi = M.macro_f1_bootstrap_ci(f)
        rows.append({"arm": arm, "method": labels[arm], "f1": M.macro_f1(f), "lo": lo, "hi": hi})
    return rows


def macro_f1_table():
    out = pd.DataFrame([{"method": r["method"], "macro-F1 [95% bootstrap CI]":
                         M.fmt_ci(r["f1"], r["lo"], r["hi"])} for r in macro_f1_rows()])
    return write_table(out, "aita_macro_f1")


def elephant_rows(env):
    labels = method_labels("Qwen3-8B")
    rows = []
    for arm in ELEPHANT_ARMS[env]:
        f = load_arm("qwen3-8b", env, arm)
        rec = {"arm": arm, "method": labels[arm]}
        for metric in ("framing", "validation"):
            if env == "AITA-NTA-FLIP":
                p, lo, hi, n = M.binary_label_rate_flip(f, metric)
            else:
                p, lo, hi, n = M.binary_label_rate(f, f"{metric}_response")
            rec[metric] = (p, lo, hi)
            rec[f"n_{metric}"] = n
        rows.append(rec)
    return rows


def elephant_table(env):
    out = pd.DataFrame([{
        "method": r["method"],
        "accepts framing [95% CI]": M.fmt_ci(*r["framing"]), "n framing": r["n_framing"],
        "validates emotions [95% CI]": M.fmt_ci(*r["validation"]),
        "n validation": r["n_validation"]} for r in elephant_rows(env)])
    return write_table(out, f"elephant_{env}")


def v1_rows(env):
    """Endorsement rate under the ORIGINAL Cheng et al. judge prompt (v1),
    with its Wald CI (the convention of the original v1 table), next to the
    modified (v5) judge's Wilson CI."""
    labels = method_labels("Qwen3-8B")
    arms = [a for a in ENDORSE_ARMS_8B[env]]
    rows = []
    for arm in arms:
        f = load_arm("qwen3-8b", env, arm)
        g = M.gated(f)
        v1 = g["endorse_v1_response"]
        assert v1.notna().all(), f"{env}/{arm}: missing v1 labels"
        n01 = int(v1.isin([0, 1]).sum())
        k = int((v1 == 1).sum())
        c = M.endorsement_counts(f)
        rows.append({"arm": arm, "method": labels[arm], "v1": M.wald(k, n01), "n01_v1": n01,
                     "v5": M.wilson(c["n1"], c["n01"]), "n_scored": len(g)})
    return rows


def v1_table(env):
    out = pd.DataFrame([{"method": r["method"],
                         "endorsement rate, original judge [Wald 95% CI]": M.fmt_ci(*r["v1"]),
                         "n_endorse (original judge)": r["n01_v1"],
                         "endorsement rate, our judge [Wilson 95% CI]": M.fmt_ci(*r["v5"]),
                         "n_scored": r["n_scored"]} for r in v1_rows(env)])
    return write_table(out, f"endorsement_v1judge_{env}")


def main():
    for env in ("OEQ", "PAS"):
        endorsement_table("qwen3-8b", env, ENDORSE_ARMS_8B[env])
        neutral_table(env)
        v1_table(env)
    explicit_stance_table()
    for env in ("AITA", "AITA-NTA-FLIP"):
        verdict_table("qwen3-8b", env)
    macro_f1_table()
    for env in ("OEQ", "PAS", "AITA", "AITA-NTA-FLIP"):
        elephant_table(env)


if __name__ == "__main__":
    main()
