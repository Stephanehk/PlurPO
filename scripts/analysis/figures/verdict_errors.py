"""AITA / AITA-Flipped verdict-error scatter plots (Fig. 3).

AITA: x = false positive rate (over-criticalness), y = false negative rate
(sycophancy), YTA as the positive class. AITA-Flipped: x = both-YTA rate,
y = both-NTA rate. Wilson 95% CIs on both axes. Numbers come from results/ via
tables_main.verdict_rows (the original figure scripts had these values typed
in).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import r3  # noqa: E402
from tables_main import verdict_rows  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

FS = 1.0
MS = 1.0
VS = 1.0

BG = "#FFFFFF"
TEXT = "#111111"
ARM_COLORS = {"base": "#E2694A", "explicit": "#2CA98F", "sharma": "#EFA61C",
              "dontbesyco": "#F2A6C4", "becritical": "#1E7F3C", "perspshift": "#4B3E9E",
              "plurpo": "#3568C4"}

# Set per call by build(); make_scatter reads them (as in the original script).
COLORS = {}
ORDER = []


def build(model, env):
    """-> rows {label: (x, xlo, xhi, y, ylo, yhi)} and sets COLORS / ORDER."""
    global COLORS, ORDER
    rows, COLORS, ORDER = {}, {}, []
    for r in verdict_rows(model, env):
        rows[r["method"]] = tuple(r3(v) for v in tuple(r["x"]) + tuple(r["y"]))
        COLORS[r["method"]] = ARM_COLORS[r["arm"]]
        ORDER.append(r["method"])
    return rows


def make_scatter(rows, xlabel, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9.3, 7.7), facecolor=BG)
    ax.set_facecolor(BG)
    for label in reversed(ORDER):
        x, xlo, xhi, y, ylo, yhi = rows[label]
        c = COLORS[label]
        ax.plot([xlo, xhi], [y, y], color=c, lw=1.5*MS, ls=(0, (4, 2)), zorder=2)
        ax.plot([x, x], [ylo, yhi], color=c, lw=1.5*MS, ls=(0, (4, 2)), zorder=2)
        ax.plot(x, y, marker='o', mfc=c, mec=c, ms=15*MS, zorder=3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax.set_xlabel(xlabel, fontsize=19*FS, color=TEXT, labelpad=8*FS)
    ax.set_ylabel(ylabel, fontsize=19*FS, color=TEXT, labelpad=8*FS)
    ax.tick_params(labelsize=15*FS, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [Line2D([0], [0], marker='o', color='none', mfc=COLORS[l], mec=COLORS[l], ms=15*MS)
               for l in ORDER]
    leg = ax.legend(handles, ORDER, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False,
                     fontsize=16*FS, handletextpad=0.5, labelspacing=0.85)
    fig.subplots_adjust(left=0.115, top=0.995, bottom=0.09)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.06)
    print("wrote", out_path)


