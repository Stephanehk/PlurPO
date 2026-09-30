"""Render every paper figure from results/ into results/_reproduced/figures/.

Output file names equal the paper's \\includegraphics targets, so the directory
can be dropped into the paper's images/. Page widths / base font sizes / dpi
per figure match the published layout (see _render.py for the font-size rule).

Run: python scripts/analysis/figures/render_all.py [--only NAME_SUBSTRING]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from common import FIG_DIR, ENDORSE_ARMS_8B, ensure_out  # noqa: E402
import _render as R  # noqa: E402
import aita_f1_samples  # noqa: E402
import elephant  # noqa: E402
import endorsement  # noqa: E402
import v1_judge  # noqa: E402
import verdict_errors  # noqa: E402

TW = R.TEXTWIDTH_IN


def jobs():
    """-> list of (file name, page width, base pt, dpi, module, draw, kwargs)."""
    out = []
    for env in ("OEQ", "PAS"):
        e = env.lower()
        rows8 = endorsement.build_rows("qwen3-8b", env, ENDORSE_ARMS_8B[env])
        out.append((f"{e}_endorsement_vs_samples_endorsement_only.png", 0.48 * TW, 15.5, 200,
                    endorsement, lambda p, r=rows8: endorsement.make_fig(r, p),
                    dict(crop_panel=True, fixed_fs=1.0)))
        out.append((f"{e}_endorsement_vs_samples.png", 0.8 * TW, 15.5, 200,
                    endorsement, lambda p, r=rows8: endorsement.make_fig(r, p), {}))
    for env, stem, xl, yl in (("AITA", "aita", "False positive rate", "False negative rate"),
                              ("AITA-NTA-FLIP", "flip", "Both-YTA rate", "Both-NTA rate")):
        def draw(p, e=env, x=xl, y=yl):
            rows = verdict_errors.build("qwen3-8b", e)
            verdict_errors.make_scatter(rows, x, y, p)
        out.append((f"{stem}_verdict_errors.png", 0.48 * TW, 16, 200,
                    verdict_errors, draw, {}))
    for env, stem, w in (("OEQ", "oeq", 0.49), ("PAS", "pas", 0.8), ("AITA", "aita", 0.49),
                         ("AITA-NTA-FLIP", "flip", 0.49)):
        out.append((f"{stem}_elephant_judges.png", w * TW, 15, 200, elephant,
                    lambda p, e=env: elephant.two_col_bar(e, p), {}))
    out.append(("aita_f1.png", 0.7 * TW, 12.5, 200, aita_f1_samples,
                aita_f1_samples.make_aita_f1, {}))
    for env, stem in (("AITA", "aita"), ("AITA-NTA-FLIP", "flip")):
        out.append((f"{stem}_samples.png", 0.48 * TW, 12.5, 200, aita_f1_samples,
                    lambda p, e=env: aita_f1_samples.make_samples_plot(e, p), {}))
    for env in ("OEQ", "PAS"):
        out.append((f"{env.lower()}_endorsement_v1judge.png", 0.8 * TW, 15.5, 200, v1_judge,
                    lambda p, e=env: v1_judge.make_fig(v1_judge.build_rows(e), p), {}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="render only names containing this")
    args = ap.parse_args()
    ensure_out()
    import matplotlib.pyplot as plt
    for name, page_w, base_pt, dpi, mod, draw, kw in jobs():
        if args.only and args.only not in name:
            continue
        def closing(p, d=draw):
            # every figure starts from the same matplotlib state (serif text)
            plt.rcdefaults()
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
            d(p)
            plt.close("all")
        R.emit(os.path.join(FIG_DIR, name), page_w, base_pt, dpi, mod, closing, **kw)


if __name__ == "__main__":
    main()
