"""Action-endorsement rate + rejection-sampling figures (OEQ / PAS).

  {oeq,pas}_endorsement_vs_samples.png         two panels (App. full figures)
  {oeq,pas}_endorsement_vs_samples_endorsement_only.png   left panel (Fig. 2)

Left panel: action endorsement rate with Wilson 95% CI. Right panel: mean
number of samples needed to pass the rejection-sampling filter, 95% normal CI
on the mean, and the number of prompts dropped after 5 failed attempts. The
OEQ human reference was never gated, so it has no samples bar. Every number
comes from results/ via tables_main.endorsement_rows.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import r3  # noqa: E402
from tables_main import endorsement_rows  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

FS = 1.0   # text-size multiplier (set by _render.emit)
MS = 1.0   # mark-size / line-width multiplier
VS = 1.0   # figure-height multiplier

BG = "#FFFFFF"
HILITE = "#E9EFF9"
BLUE = "#3568C4"
GRAY_DOT = "#6E6E6E"
GRAY_BAR = "#9A9A9A"
TEXT = "#111111"


def build_rows(model, env, arms, samples_ci=True, relabel=None):
    """Rows in the plotting schema. `samples_ci=False` omits the samples error
    bar (the published Qwen3-32B figures were drawn without one); `relabel`
    maps arm key -> display label overrides."""
    out = []
    for r in endorsement_rows(model, env, arms):
        if relabel and r["arm"] in relabel:
            r["method"] = relabel[r["arm"]]
        ci = None
        if samples_ci and r["mean_samples"] is not None:
            ci = (r["samples_lo"], r["samples_hi"])
        out.append(dict(method=r["method"], endorsement_p=r3(r["rate"]),
                        endorsement_lo=r3(r["lo"]), endorsement_hi=r3(r["hi"]), mean_samples=r["mean_samples"],
                        n_filtered=r["n_filtered"], samples_ci=ci))
    return out


def make_fig(rows, out_path, highlight_label="PlurPO",
             samples_ticks=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0), samples_fmt="{:.3f}"):
    n = len(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, (0.58 * n + 0.35) * VS), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    y = list(range(n, 0, -1))

    for yi, r in zip(y, rows):
        is_hi = r["method"] == highlight_label
        color = BLUE if is_hi else GRAY_DOT
        if is_hi:
            ax1.axhspan(yi - 0.5, yi + 0.5, color=HILITE, zorder=0)
            ax2.axhspan(yi - 0.5, yi + 0.5, color=HILITE, zorder=0)

        if r["endorsement_p"] is not None:
            p, lo, hi = r["endorsement_p"], r["endorsement_lo"], r["endorsement_hi"]
            ax1.plot([lo, hi], [yi, yi], color=color, lw=1.6*MS, ls=(0, (4, 2)), zorder=2)
            if r["method"] == "Human baseline":
                ax1.plot(p, yi, marker='o', mfc='white', mec=GRAY_DOT, mew=1.8*MS, ms=11*MS, zorder=3)
            else:
                ax1.plot(p, yi, marker='o', mfc=color, mec=color, ms=11*MS, zorder=3)
            ax1.text(1.04, yi, f"{p*100:.1f}%", transform=ax1.get_yaxis_transform(),
                      va='center', ha='left', fontsize=15*FS, color=TEXT)

        ms = r["mean_samples"]
        if ms is not None:
            ax2.barh(yi, ms - 1.0, left=1.0, height=0.38,
                      color=BLUE if is_hi else GRAY_BAR, zorder=2)
            if r["samples_ci"] is not None:
                lo, hi = r["samples_ci"]
                ax2.plot([lo, hi], [yi, yi], color="#333333", lw=1.6*MS, zorder=3)
                ax2.plot([lo, lo], [yi - 0.09, yi + 0.09], color="#333333", lw=1.6*MS, zorder=3)
                ax2.plot([hi, hi], [yi - 0.09, yi + 0.09], color="#333333", lw=1.6*MS, zorder=3)
            nf = r["n_filtered"]
            ax2.text(1.06, yi, samples_fmt.format(ms), transform=ax2.get_yaxis_transform(),
                      va='center', ha='left', fontsize=15*FS, color=TEXT)
            if isinstance(nf, int) and nf > 0:
                ax2.text(1.28, yi, f"{nf} dropped", transform=ax2.get_yaxis_transform(),
                          va='center', ha='left', fontsize=13*FS, style='italic', color="#888888")

    for ax in (ax1, ax2):
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(0.42, n + 0.42)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False)

    ax1.set_xlim(0, 1.0)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13*FS, color=TEXT)
    ax1.grid(axis='x', color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax1.set_xlabel("Endorsement rate", fontsize=17*FS, color=TEXT, labelpad=8*FS)

    _tk = list(samples_ticks)
    _tkfmt = "{:.2f}" if len(_tk) > 6 else "{:.1f}"
    ax2.set_xlim(_tk[0], _tk[-1])
    ax2.set_xticks(_tk)
    ax2.set_xticklabels([_tkfmt.format(v) for v in _tk], fontsize=13*FS, color=TEXT)
    ax2.grid(axis='x', color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax2.set_xlabel("Mean samples to produce judged response", fontsize=17*FS, color=TEXT, labelpad=8*FS)

    fig.subplots_adjust(left=0.29, right=0.86, top=0.995, bottom=0.115, wspace=0.36)

    for yi, r in zip(y, rows):
        w = 'bold' if r["method"] == highlight_label else 'normal'
        ax1.annotate(r["method"], xy=(0, yi), xycoords=('axes fraction', 'data'),
                     xytext=(-8, 0), textcoords='offset points',
                     ha='right', va='center', fontsize=15.5*FS, fontweight=w, color=TEXT)

    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.06)
    print("wrote", out_path)


