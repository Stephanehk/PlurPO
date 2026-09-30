"""AITA macro-F1 figure and AITA / AITA-Flipped rejection-sampling figures
(App. "AITA Macro F1 Scores" and "Rejection sampling results for AITA and
AITA-Flipped").

Macro-F1 = mean of F1 with YTA positive and F1 with NTA positive; its CI is a
joint percentile bootstrap (3000 resamples). Mean samples: 95% normal CI on
the mean over every scored key (FLIP: both sides of every surviving pair).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tables_main import macro_f1_rows, verdict_rows  # noqa: E402
from verdict_errors import ARM_COLORS  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

FS = 1.0
MS = 1.0
VS = 1.0

BG = "#FFFFFF"
TEXT = "#111111"


def f1_data():
    rows = macro_f1_rows()
    return ([r["method"] for r in rows], {r["method"]: r["f1"] for r in rows},
            {r["method"]: (r["lo"], r["hi"]) for r in rows},
            {r["method"]: ARM_COLORS[r["arm"]] for r in rows})


def samples_data(env):
    rows = verdict_rows("qwen3-8b", env)
    return ([r["method"] for r in rows],
            {r["method"]: dict(n_filtered=r["n_filtered"], mean=r["mean_samples"],
                               lo=r["samples_lo"], hi=r["samples_hi"]) for r in rows},
            {r["method"]: ARM_COLORS[r["arm"]] for r in rows})


def samples_panel(ax, order, gate, colors, show_labels=True):
    n = len(order)
    y = list(range(n, 0, -1))
    for yi, label in zip(y, order):
        g = gate[label]
        ms, lo, hi = g["mean"], g["lo"], g["hi"]
        c = colors[label]
        ax.barh(yi, ms - 1.0, left=1.0, height=0.5, color=c, zorder=2)
        ax.plot([lo, hi], [yi, yi], color="#333333", lw=1.6*MS, zorder=3)
        ax.plot([lo, lo], [yi - 0.09, yi + 0.09], color="#333333", lw=1.6*MS, zorder=3)
        ax.plot([hi, hi], [yi - 0.09, yi + 0.09], color="#333333", lw=1.6*MS, zorder=3)
        ax.text(1.06, yi, f"{ms:.3f}", transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=13*FS, color=TEXT, zorder=3)
        if g["n_filtered"]:
            ax.text(1.30, yi, f"{g['n_filtered']} dropped", transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=11.5*FS, style='italic', color="#888888", zorder=3)
        if show_labels:
            ax.annotate(label, xy=(0, yi), xycoords=('axes fraction', 'data'),
                        xytext=(-8, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=12.5*FS, color=TEXT)
    ax.set_xlim(1.0, 2.0)
    ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_xticklabels([f"{v:.1f}" for v in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]])
    ax.set_ylim(0.4, n + 0.6)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, labelsize=12*FS, colors=TEXT)
    ax.grid(axis='x', color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax.set_xlabel("Mean samples to produce judged response", fontsize=15.5*FS, color=TEXT, labelpad=8*FS)


def f1_panel(ax, order, f1, f1_ci, colors, show_labels=True):
    n = len(order)
    y = list(range(n, 0, -1))
    for yi, label in zip(y, order):
        c = colors[label]
        v = f1[label]
        lo, hi = f1_ci[label]
        ax.barh(yi, v, height=0.5, color=c, zorder=2)
        ax.plot([lo, hi], [yi, yi], color="#333333", lw=1.6*MS, zorder=3)
        ax.plot([lo, lo], [yi - 0.09, yi + 0.09], color="#333333", lw=1.6*MS, zorder=3)
        ax.plot([hi, hi], [yi - 0.09, yi + 0.09], color="#333333", lw=1.6*MS, zorder=3)
        ax.text(hi + 0.03, yi, f"{v:.3f}", va='center', ha='left', fontsize=13*FS, color=TEXT, zorder=3)
        if show_labels:
            ax.annotate(label, xy=(0, yi), xycoords=('axes fraction', 'data'),
                        xytext=(-8, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=12.5*FS, color=TEXT)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.4, n + 0.6)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, labelsize=13*FS, colors=TEXT)
    ax.grid(axis='x', color="#DDDDDD", lw=0.8*MS, zorder=0)
    ax.set_xlabel("Macro F1 Score", fontsize=17*FS, color=TEXT, labelpad=8*FS)


def make_aita_f1(out_path):
    order, f1, ci, colors = f1_data()
    fig, ax = plt.subplots(figsize=(9.5, 7.4 * VS), facecolor=BG)
    ax.set_facecolor(BG)
    f1_panel(ax, order, f1, ci, colors)
    fig.subplots_adjust(left=0.32, right=0.85, top=0.97, bottom=0.1)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def make_samples_plot(env, out_path):
    order, gate, colors = samples_data(env)
    fig, ax = plt.subplots(figsize=(9.5, 7.4 * VS), facecolor=BG)
    ax.set_facecolor(BG)
    samples_panel(ax, order, gate, colors)
    fig.subplots_adjust(left=0.32, right=0.8, top=0.97, bottom=0.1)
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
