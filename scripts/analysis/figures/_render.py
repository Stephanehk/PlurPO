"""Figure layout helpers: print-size-matched font scaling and panel cropping.

Ported from the scripts that rendered the paper's figures.

FONT TARGET. Figure 2 (oeq_endorsement_vs_samples_endorsement_only.png) is
rendered 1661 px wide at 200 dpi and placed at 0.48\\textwidth; its method
names (15.5 pt in the source) therefore print at 4.927 pt. Every other figure
gets one uniform font multiplier FS, solved so its method-name text prints at
that same size once LaTeX scales it to its \\includegraphics width (FS >= 1).
MS scales marks/line widths with the figure's width growth so marks keep their
printed size; VS = FS grows row plots' height with the text.

CROP. `*_endorsement_only.png` are cut from the two-panel figure: left = first
non-white column - 4, right = start of the widest white run in the gutter + 5.
"""

import os
import shutil

import numpy as np
from PIL import Image

TEXTWIDTH_IN = 5.5                      # paper \textwidth: 5.5 in
FIG2_SRC_WIDTH_IN = 1661 / 200.0        # oeq_..._endorsement_only.png at 200 dpi
FIG2_PAGE_WIDTH_IN = 0.48 * TEXTWIDTH_IN
FIG2_LABEL_PT = 15.5                    # method-name labels in plot_endorsement_vs_samples.py
TARGET_PT = FIG2_LABEL_PT * FIG2_PAGE_WIDTH_IN / FIG2_SRC_WIDTH_IN   # 4.9272 pt


def printed_pt(base_label_pt, fs, page_width_in, w_px, dpi):
    """Size, in points on the printed page, of text set at `base_label_pt*fs`
    in a figure `w_px` wide that LaTeX scales to `page_width_in`."""
    return base_label_pt * fs * page_width_in / (w_px / dpi)


def solve_fs(render, page_width_in, base_label_pt, dpi, tag="", tol_rel=0.005,
             probe=1.6, max_iter=12, verbose=True):
    """Find the uniform font multiplier FS so the figure's method-name text
    prints at TARGET_PT.

    `render(fs)` draws+saves the figure at multiplier `fs` and returns the
    final artifact's pixel width. printed(fs) = base*fs*page*dpi / w(fs) is
    monotone increasing and saturating (w grows roughly affinely in fs), so a
    secant iteration on printed(fs) - TARGET converges in a few renders.

    FS is clamped at >= 1.0 (never shrink), per the user's decision.
    Returns (fs, width_px, printed_pt).
    """
    def P(fs, w):
        return printed_pt(base_label_pt, fs, page_width_in, w, dpi)

    w1 = render(1.0)
    p1 = P(1.0, w1)
    if verbose:
        print(f"    [{tag}] FS=1.000 -> {w1}px, prints {p1:.3f}pt (target {TARGET_PT:.3f})")
    if p1 >= TARGET_PT:
        if verbose:
            print(f"    [{tag}] already >= target; fonts left unchanged (FS=1.0)")
        return 1.0, w1, p1

    # closed-form first guess from an affine fit of w(fs) through two probes
    w2 = render(probe)
    p2 = P(probe, w2)
    B = (w2 - w1) / (probe - 1.0)
    A = w1 - B
    denom = base_label_pt * page_width_in * dpi - TARGET_PT * B
    assert denom > 0, (f"{tag}: target unreachable -- printed size saturates at "
                       f"{base_label_pt * page_width_in * dpi / B:.2f}pt")
    x0, f0 = probe, p2 - TARGET_PT
    x1 = TARGET_PT * A / denom
    for _ in range(max_iter):
        w = render(x1)
        f1 = P(x1, w) - TARGET_PT
        if verbose:
            print(f"    [{tag}] FS={x1:.4f} -> {w}px, prints {f1 + TARGET_PT:.3f}pt")
        if abs(f1) / TARGET_PT < tol_rel:
            return x1, w, f1 + TARGET_PT
        if f1 == f0:
            break
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)   # secant
    raise AssertionError(f"solve_fs did not converge for {tag}")

GRID_RGB = (221, 221, 221)


def _cols(im):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    h = a.shape[0]
    nonwhite = (a < 250).any(axis=2).any(axis=0)
    grid = (np.abs(a - np.array(GRID_RGB, dtype=np.int16)).sum(2) < 6).sum(0) > 0.5 * h
    return nonwhite, np.nonzero(grid)[0]


def crop_bounds(im, gap=200):
    nonwhite, gcols = _cols(im)
    assert len(gcols) >= 4, "no x-grid found; is this the two-panel figure?"
    breaks = np.nonzero(np.diff(gcols) > gap)[0]
    assert len(breaks) == 1, f"expected exactly one panel gap, found {len(breaks)}"
    panel2_left = int(gcols[breaks[0] + 1])
    panel1_right = int(gcols[breaks[0]])
    # widest all-white column run in the gutter between the two panels
    white = ~nonwhite
    best, run_start, cur = (0, None), None, 0
    for c in range(panel1_right + 1, panel2_left):
        if white[c]:
            if cur == 0:
                run_start = c
            cur += 1
            if cur > best[0]:
                best = (cur, run_start)
        else:
            cur = 0
    assert best[1] is not None, "no white gutter between panels"
    nz = np.nonzero(nonwhite)[0]
    left = max(0, int(nz[0]) - 4)
    right = int(best[1]) + 5
    return left, right


def crop(in_path, out_path):
    """Crop the left (endorsement) panel of a two-panel figure."""
    im = Image.open(in_path)
    left, right = crop_bounds(im)
    out = im.crop((left, 0, right, im.size[1]))
    out.save(out_path)
    return out.size


def emit(path, page_w, base_pt, dpi, mod, draw, crop_panel=False, fixed_fs=None):
    """Render `draw(tmp_path)` so method-name text prints at TARGET_PT.

    `mod` is the plotting module whose FS / MS / VS globals `draw` reads.
    `page_w` is the figure's \\includegraphics width in inches and `base_pt`
    the source point size of its method-name text. Returns the solved FS."""
    tmp = path + ".tmp.png"

    def _draw(fs, ms):
        mod.FS, mod.MS, mod.VS = fs, ms, fs
        draw(tmp)
        if crop_panel:
            w = crop(tmp, path)[0]
        else:
            shutil.copyfile(tmp, path)
            w = Image.open(path).size[0]
        return w

    w_orig = _draw(1.0, 1.0)
    last = {"w": w_orig}

    def render(fs):
        w = _draw(fs, last["w"] / w_orig)
        last["w"] = w
        return w

    if fixed_fs is not None:
        render(fixed_fs)
        fs = fixed_fs
    else:
        fs, _w, _p = solve_fs(render, page_w, base_pt, dpi, tag=os.path.basename(path),
                              verbose=False)
    os.remove(tmp)
    print(f"[figure] {path} (FS={fs:.3f})")
    return fs
