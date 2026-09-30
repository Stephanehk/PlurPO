"""ELEPHANT framing / validation judge figures (Fig. 4 top for PAS; App. figure
for OEQ, AITA, AITA-Flipped): share of responses that accept the user's framing
and share that validate the user's emotions, Wilson 95% CIs, over
rejection-sampled responses. Numbers come from tables_main.elephant_rows.

Note: as published, these figures' "DPO-Neutral" row is the non-mixed Neutral
DPO run (common.ELEPHANT_ARMS), unlike the endorsement figures.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import r3  # noqa: E402
from tables_main import elephant_rows  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

FS = 1.0
MS = 1.0
VS = 1.0

BG = "#FFFFFF"
TEXT = "#111111"
HILITE = "#E9EFF9"
BLUE = "#3568C4"
ORANGE = "#D9782D"


def two_col_bar(env, out_path, highlight="PlurPO"):
    rows = elephant_rows(env)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(15, (0.62 * n + 1.1) * VS), facecolor=BG)
    ax.set_facecolor(BG)
    y = list(range(n, 0, -1))
    bar_h = 0.36
    for yi, row in zip(y, rows):
        m = row["method"]
        is_hi = m == highlight
        if is_hi:
            ax.axhspan(yi - 0.5, yi + 0.5, color=HILITE, zorder=0)
        p1, lo1, hi1 = (r3(v) for v in row["framing"])
        p2, lo2, hi2 = (r3(v) for v in row["validation"])
        ax.barh(yi + bar_h/2 + 0.02, p1, height=bar_h, color=BLUE, zorder=2)
        ax.plot([lo1, hi1], [yi + bar_h/2 + 0.02]*2, color="#555555", lw=1.3*MS, zorder=3)
        ax.barh(yi - bar_h/2 - 0.02, p2, height=bar_h, color=ORANGE, zorder=2)
        ax.plot([lo2, hi2], [yi - bar_h/2 - 0.02]*2, color="#555555", lw=1.3*MS, zorder=3)
        ax.text(1.02, yi + bar_h/2 + 0.02, f"{p1*100:.1f}%", va='center', ha='left', fontsize=13.5*FS, color=TEXT)
        ax.text(1.02, yi - bar_h/2 - 0.02, f"{p2*100:.1f}%", va='center', ha='left', fontsize=13.5*FS, color=TEXT)
        w = 'bold' if is_hi else 'normal'
        ax.annotate(m, xy=(0, yi), xycoords=('axes fraction', 'data'), xytext=(-8, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=15*FS, fontweight=w, color=TEXT)
    ax.set_ylim(0.4, n + 0.6)
    ax.set_xlim(0, 1.0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13.5*FS, color=TEXT)
    ax.grid(axis='x', color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax.set_xlabel("Share of responses", fontsize=18*FS, color=TEXT, labelpad=8*FS)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Accepts user's framing", "Validates user's emotions"], loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=15.5*FS, handlelength=1.2)
    fig.subplots_adjust(left=0.32, right=0.98, top=0.93, bottom=0.09)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
