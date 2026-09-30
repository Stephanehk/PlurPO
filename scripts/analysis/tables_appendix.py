"""Appendix tables, computed from results/ only.

  veto_vs_endorse              Tab. veto-vs-endorse-by-epoch (Wilson 95% CIs)

The other appendix tables (per-epoch, alpha sweep, role confusion,
Inferred-Prefs-DPO, in-context, ablations, reliability, IFEval) were computed
from first-draw / reliability / IFEval outputs that are not included in this
release.

Run: python scripts/analysis/tables_appendix.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_table  # noqa: E402
from plurpo import metrics as M  # noqa: E402
from plurpo.paths import RESULTS_DIR  # noqa: E402


def veto_vs_endorse():
    rows = []
    for it in (1, 2, 3):
        d = pd.read_csv(os.path.join(RESULTS_DIR, "veto_vs_endorse", f"iter{it}_labels.csv"))
        lab = d["endorse_response"].astype(int)
        vet = d["vetoed"].astype(bool)
        rec = {"epoch": it}
        # "endorse" pools explicit (1) and implicit (2) endorsement, as in the paper.
        for name, codes in (("challenge", (0,)), ("endorse", (1, 2))):
            sub = vet[lab.isin(codes)]
            p, lo, hi = M.wilson(int(sub.sum()), len(sub))
            rec[f"P(veto|{name})"] = f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
            rec[f"n {name}"] = len(sub)
        rows.append(rec)
    return write_table(pd.DataFrame(rows), "veto_vs_endorse")


def main():
    veto_vs_endorse()


if __name__ == "__main__":
    main()
