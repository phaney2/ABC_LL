"""Shared plotting style.

Colours are the unmodified reference data-viz palette.  Only categorical slots
1-3 are used: that subset is validated for *all* pairs (not just adjacent) in
light mode, which is what a chart with freely-overlapping marks needs.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# categorical slots 1-3, light mode, in fixed order -- never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# sequential blue ramp, for magnitude (heatmaps)
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]


#: Second sequential ramp, for when two magnitude scales appear side by side.
#: One hue (the categorical orange, slot 2), light -> dark.
SEQ_ORANGE = ["#fdeee6", "#fbd9c7", "#f8bc9c", "#f49a6c", "#ef7c45",
              "#eb6834", "#d4551f", "#b04418", "#8a3412", "#63240c"]


def sequential_cmap(name="abc_blue", hue="blue"):
    from matplotlib.colors import LinearSegmentedColormap

    steps = {"blue": SEQ_BLUE, "orange": SEQ_ORANGE}[hue]
    return LinearSegmentedColormap.from_list(name, steps)


def apply():
    """Install the style globally."""
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 1.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
    })


def _align(a, b, gap):
    """Order-preserving minimum-cost alignment of two sorted level arrays.

    Returns a list of ``(i, j)`` index pairs.  Levels never change their
    relative order as a parameter is swept, so the matching between adjacent
    columns must be monotonic -- which is exactly a global sequence alignment
    with cost ``|a_i - b_j|`` and a constant penalty ``gap`` for a level that
    appears or disappears (at the edges of the energy window).
    """
    na, nb = len(a), len(b)
    d = np.empty((na + 1, nb + 1))
    d[0, :] = gap * np.arange(nb + 1)
    d[:, 0] = gap * np.arange(na + 1)
    for i in range(1, na + 1):
        sub = d[i - 1, :nb] + np.abs(a[i - 1] - b)
        row = d[i]
        for j in range(1, nb + 1):
            row[j] = min(sub[j - 1], d[i - 1, j] + gap, row[j - 1] + gap)
    pairs, i, j = [], na, nb
    while i > 0 and j > 0:
        if np.isclose(d[i, j], d[i - 1, j - 1] + abs(a[i - 1] - b[j - 1])):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif np.isclose(d[i, j], d[i - 1, j] + gap):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def trace_branches(levels_by_x, gap=None):
    """Follow each Landau level continuously across a parameter sweep.

    ``branches`` below indexes levels outward from E = 0, which breaks
    wherever a level crosses zero (every index after it shifts by one, and the
    line jumps).  This instead matches levels between adjacent columns by
    proximity, so each row of the result is one physical level.

    Returns an array of shape ``(n_branch, n_x)`` padded with NaN.
    """
    cols = [np.sort(np.asarray(e)) for e in levels_by_x]
    if gap is None:
        spread = max((np.ptp(c) for c in cols if c.size), default=1.0)
        gap = 0.05 * spread

    rows = [[v] for v in cols[0]]          # branch -> list of values so far
    active = list(range(len(cols[0])))     # branch id for each level in prev col
    for j in range(1, len(cols)):
        prev, cur = cols[j - 1], cols[j]
        nxt = [None] * len(cur)
        for ia, ib in _align(prev, cur, gap):
            nxt[ib] = active[ia]
        for ib, bid in enumerate(nxt):
            if bid is None:                # a level entered the window
                rows.append([np.nan] * j)
                nxt[ib] = len(rows) - 1
        for ib, bid in enumerate(nxt):
            rows[bid].append(cur[ib])
        for bid in range(len(rows)):       # levels that left the window
            if len(rows[bid]) == j:
                rows[bid].append(np.nan)
        active = nxt

    return np.array([r + [np.nan] * (len(cols) - len(r)) for r in rows])


def branches(levels_by_x):
    """Reorganise a list of per-x level arrays into continuous branches.

    Levels are indexed outward from E = 0 (conduction: 0, 1, 2, ...; valence:
    -1, -2, ...).  Because the index ordering is monotonic in energy, branches
    never cross, so each can be drawn as a single line.  Missing entries are
    padded with NaN.

    Returns ``(conduction, valence)``, each of shape ``(n_branch, n_x)``.
    """
    up, dn = [], []
    for e in levels_by_x:
        e = np.asarray(e)
        up.append(np.sort(e[e > 0]))
        dn.append(np.sort(e[e <= 0])[::-1])

    def pad(seqs):
        m = max((len(s) for s in seqs), default=0)
        out = np.full((m, len(seqs)), np.nan)
        for j, s in enumerate(seqs):
            out[: len(s), j] = s
        return out

    return pad(up), pad(dn)


def label_line(ax, x, y, text, color, **kw):
    """Direct label at the end of a line (the relief rule for low-contrast hues)."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8, va="center", fontweight="semibold", **kw)
