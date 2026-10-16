"""Export raw Landau levels (and their layer weights) to .mat for MATLAB.

Writes three files into data/:

  ll_vs_B.mat      levels on a B grid at a few fixed dU
  ll_vs_dU.mat     levels on a dU grid (both signs) at a few fixed B
  ll_grid.mat      levels on a coarse 2-D (B, dU) grid

Each is NaN-padded along the level axis with an ``n_levels`` count, carries
``layer_weight`` (fraction on [bottom, middle, top]), and is self-describing
via its ``units`` and ``readme`` fields.  Valleys are merged and sorted by
energy; ``valley`` records which valley each level came from.

Run:  python3 scripts/export_levels.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import dos as dosmod, landau, matio  # noqa: E402
from abc_ll.params import NPHI_1T  # noqa: E402
from abc_ll.params import DEFAULT as P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

MODEL = "six_band"
WINDOW = 0.13          # eV, symmetric: covers both electron and hole sides


def _solve(args):
    """Both valleys at one (B, dU), merged and sorted, with layer weights."""
    B, du = args
    es, ws, vs = [], [], []
    for xi in (+1, -1):
        lv = landau.solve_auto(MODEL, B, du, P, valley=xi, window=WINDOW,
                               layers=True)
        es.append(lv.energy)
        ws.append(lv.layer_weight)
        vs.append(np.full(len(lv), xi))
    e = np.concatenate(es)
    w = np.concatenate(ws, axis=0)
    v = np.concatenate(vs)
    o = np.argsort(e)
    return e[o], w[o], v[o]


def run(B, du, path, note=""):
    pts = [(b, d) for b in B for d in du]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_solve, pts, chunksize=4))
    nB, nU = len(B), len(du)
    levels = [[res[i * nU + j][0] for j in range(nU)] for i in range(nB)]
    weights = [[res[i * nU + j][1] for j in range(nU)] for i in range(nB)]
    counts = np.array([[len(levels[i][j]) for j in range(nU)]
                       for i in range(nB)])
    nmax = int(counts.max())
    valley = np.full((nB, nU, nmax), np.nan)
    for i in range(nB):
        for j in range(nU):
            valley[i, j, :counts[i, j]] = res[i * nU + j][2]

    matio.save_levels(path, B, du, levels, weights, P, MODEL,
                      extra={"valley": valley,
                             "energy_window_eV": WINDOW,
                             "note": note})
    print(f"  wrote {path}   ({nB} x {nU}, up to {nmax} levels)")


def run_nu(B, nu, path):
    """Levels on a rectangular (B, filling) grid.

    dU is not a free axis here: a single gate ties it to the filling by
    dU = nu*(e*B/h)/kappa, so it varies across the grid.
    """
    gate = dosmod.moire_gate()
    du2d = nu[None, :] * (NPHI_1T * B)[:, None] / gate.kappa
    nB, nN = len(B), len(nu)
    if np.max(np.abs(du2d)) > gate.du_max + 1e-12:
        over = np.max(np.abs(du2d))
        print(f"    note: |dU| reaches {over*1e3:.1f} meV, beyond the "
              f"{gate.du_max*1e3:.0f} meV the calibration was set at")
    pts = [(B[i], du2d[i, j]) for i in range(nB) for j in range(nN)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_solve, pts, chunksize=8))
    lv = [[res[i * nN + j][0] for j in range(nN)] for i in range(nB)]
    lw = [[res[i * nN + j][1] for j in range(nN)] for i in range(nB)]
    vv = [[res[i * nN + j][2] for j in range(nN)] for i in range(nB)]
    matio.save_levels_nu(path, B, nu, du2d, lv, lw, vv, P, gate, MODEL,
                         WINDOW)
    nmax = max(len(lv[i][j]) for i in range(nB) for j in range(nN))
    print(f"  wrote {path}   ({nB} x {nN}, up to {nmax} levels)")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)

    print("levels vs B at fixed dU")
    run(np.linspace(1.0, 15.0, 240),
        np.array([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]),
        os.path.join(CACHE, "ll_vs_B.mat"),
        note="Landau fan: levels vs B at seven fixed interlayer potentials, "
             "both signs. Build the DOS(E, B) map from these in MATLAB.")

    print("levels vs dU at fixed B")
    run(np.array([2.0, 6.0, 10.0, 15.0]), np.linspace(-0.20, 0.20, 401),
        os.path.join(CACHE, "ll_vs_dU.mat"),
        note="Levels vs interlayer potential, both signs, at four fields.")

    print("levels on a coarse 2-D grid")
    run(np.linspace(1.0, 15.0, 60), np.linspace(-0.20, 0.20, 81),
        os.path.join(CACHE, "ll_grid.mat"),
        note="Coarse 2-D (B, dU) grid of levels, for slicing in MATLAB.")

    print("levels on a (B, filling) grid")
    run_nu(np.linspace(1.0, 15.0, 110), np.linspace(-14.0, 14.0, 281),
           os.path.join(CACHE, "ll_vs_nu.mat"))
