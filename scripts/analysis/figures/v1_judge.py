"""Action endorsement under the ORIGINAL Cheng et al. judge prompt (App. "Measuring
Action Endorsement via a LM judge"): foreground = original (v1) judge with its
Wald 95% CI; faded background = our modified (v5) judge with its Wilson CI.
Numbers come from tables_main.v1_rows.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import HUMAN, OURS, r3  # noqa: E402
from tables_main import v1_rows  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

FS = 1.0
MS = 1.0
VS = 1.0

BG = "#FFFFFF"
HILITE = "#E9EFF9"
BLUE = "#3568C4"
GRAY_DOT = "#6E6E6E"
TEXT = "#111111"
BG_ALPHA = 0.32  # opacity of the faded v5 (our judge) reference layer

HIGHLIGHT_METHOD = OURS
HUMAN_METHOD = HUMAN
# The published figure labels the untrained model "Qwen3-8B (base)".
DISPLAY_LABEL = {"Qwen3-8B": "Qwen3-8B (base)"}


def build_rows(env):
    rows = []
    for r in v1_rows(env):
        rows.append(dict(fig_name=r["method"],
                         v5=dict(zip(("p", "lo", "hi"), (r3(v) for v in r["v5"]))),
                         v1=dict(zip(("p", "lo", "hi"), (r3(v) for v in r["v1"])))))
    return rows


def make_fig(rows, out_path):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, (0.58 * n + 0.35) * VS), facecolor=BG)
    ax.set_facecolor(BG)

    y = list(range(n, 0, -1))

    for yi, r in zip(y, rows):
        is_hi = r["fig_name"] == HIGHLIGHT_METHOD
        is_human = r["fig_name"] == HUMAN_METHOD
        color = BLUE if is_hi else GRAY_DOT
        if is_hi:
            ax.axhspan(yi - 0.5, yi + 0.5, color=HILITE, zorder=0)

        # background: current (v5) judge, faded, drawn first / lower zorder
        d5 = r["v5"]
        if d5["p"] is not None:
            ax.plot([d5["lo"], d5["hi"]], [yi, yi], color=color, lw=1.6*MS,
                     ls=(0, (4, 2)), alpha=BG_ALPHA, zorder=1)
            if is_human:
                ax.plot(d5["p"], yi, marker='o', mfc='white', mec=color, mew=1.8*MS,
                         ms=11*MS, alpha=BG_ALPHA, zorder=1)
            else:
                ax.plot(d5["p"], yi, marker='o', mfc=color, mec=color, ms=11*MS,
                         alpha=BG_ALPHA, zorder=1)

        # foreground: original (v1) judge, full opacity, drawn on top
        d1 = r["v1"]
        if d1["p"] is not None:
            ax.plot([d1["lo"], d1["hi"]], [yi, yi], color=color, lw=1.6*MS,
                     ls=(0, (4, 2)), zorder=2)
            if is_human:
                ax.plot(d1["p"], yi, marker='o', mfc='white', mec=color, mew=1.8*MS,
                         ms=11*MS, zorder=3)
            else:
                ax.plot(d1["p"], yi, marker='o', mfc=color, mec=color, ms=11*MS, zorder=3)
            label = f"{d1['p']*100:.1f}%"
            if not d1.get("complete", True):
                label += " (partial)"
            if d5["p"] is not None:
                label += f"   (v5: {d5['p']*100:.1f}%)"
            ax.text(1.04, yi, label, transform=ax.get_yaxis_transform(),
                     va='center', ha='left', fontsize=13.5*FS, color=TEXT)

    ax.set_yticks([])
    ax.set_ylim(0.42, n + 0.42)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)

    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13*FS, color=TEXT)
    ax.grid(axis='x', color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax.set_xlabel("Endorsement rate", fontsize=17*FS, color=TEXT, labelpad=8*FS)

    for yi, r in zip(y, rows):
        w = 'bold' if r["fig_name"] == HIGHLIGHT_METHOD else 'normal'
        ax.annotate(DISPLAY_LABEL.get(r["fig_name"], r["fig_name"]), xy=(0, yi), xycoords=('axes fraction', 'data'),
                     xytext=(-8, 0), textcoords='offset points',
                     ha='right', va='center', fontsize=15.5*FS, fontweight=w, color=TEXT)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color=GRAY_DOT, mfc=GRAY_DOT, mec=GRAY_DOT,
                   lw=1.6*MS, ms=10*MS, label="Original LM judge prompt from Cheng et al. (2026b)"),
        plt.Line2D([0], [0], marker='o', color=GRAY_DOT, mfc=GRAY_DOT, mec=GRAY_DOT,
                   lw=1.6*MS, ms=10*MS, alpha=BG_ALPHA, label="Our modified LM judge prompt"),
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=False, fontsize=12.5*FS)

    fig.subplots_adjust(left=0.29, right=0.86, top=0.98, bottom=0.185)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.06)
    print("wrote", out_path)


