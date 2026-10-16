"""Low-field Landau levels of the minimal six-band model at dU = 0.

A focused export: `six_band_minimal` (gamma0 and gamma1 only), zero
displacement field, down to very low B where the fan becomes dense.

Two things make this case specific.  With gamma2 = gamma3 = gamma4 = delta
= 0 the ABC chain is bipartite *and* unshifted, so charge neutrality really
is at E = 0 here -- unlike the full model, where delta and gamma4 put it at
+6.3 meV.  And the three genuine zero modes per valley sit at exactly E = 0,
which is also where the truncation artefacts sit, so they cannot be
separated by energy: `solve_auto` must use the eigenvector tail filter
(`fast=False`), not the eigenvalue-matching shortcut.

Run:  python3 scripts/export_minimal_lowB.py
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import landau, matio  # noqa: E402
from abc_ll.params import DEFAULT as P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

MODEL = "six_band_minimal"
#: Energy window and displacement field, both overridable from the
#: environment.  The window must reach past the band edges: at dU = 0.2 eV
#: the minimal model has a 163 meV gap (edges at +-81.4 meV), so +-50 meV
#: would contain no levels at all.  Widening it is cheap here because the
#: gap is empty -- the extra range selects no extra eigenvectors.
DU = float(os.environ.get("DU", 0.0))
WINDOW = float(os.environ.get("WINDOW", 0.135 if DU else 0.05))
B_MIN, B_MAX = 0.20, 5.0
#: B sampling. The fan is smooth in B, so this only sets how finely the
#: image is rendered -- too few columns and every level reads as a
#: staircase, which looks like structure but is not.
N_B = int(os.environ.get("NB", 49))


def _solve(B):
    es, ws, vs = [], [], []
    for xi in (+1, -1):
        lv = landau.solve_auto(MODEL, B, DU, P, valley=xi, window=WINDOW,
                               layers=True, fast=False)
        es.append(lv.energy)
        ws.append(lv.layer_weight)
        vs.append(np.full(len(lv), xi))
    e = np.concatenate(es)
    o = np.argsort(e)
    return e[o], np.concatenate(ws, axis=0)[o], np.concatenate(vs)[o], \
        lv.n_ll, lv.n_dropped


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    B = np.linspace(B_MIN, B_MAX, N_B)
    print(f"{MODEL}, dU = {DU*1e3:.0f} meV, |E| < {WINDOW*1e3:.0f} meV")
    print(f"B = {B_MIN} to {B_MAX} T, {N_B} points")

    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_solve, B, chunksize=1))
    print(f"  solved in {time.perf_counter()-t0:.0f} s")

    counts = [len(r[0]) for r in res]
    print(f"  levels per field: {min(counts)} (at {B_MAX} T) "
          f"to {max(counts)} (at {B_MIN} T)")
    print(f"  basis at lowest B: n_ll = {res[0][3]}, "
          f"{res[0][4]} artefacts rejected")

    nzero = [int(np.sum(np.abs(r[0]) < 1e-9)) for r in res]
    if DU == 0.0:
        # three zero modes per valley, six in total, at every field
        print(f"  zero modes found: {sorted(set(nzero))}  (expect [6])")
    else:
        # the zero multiplet is pushed to +-dU/2 and E = 0 sits in the gap
        lo = max(r[0][r[0] < 0].max() for r in res)
        hi = min(r[0][r[0] > 0].min() for r in res)
        print(f"  smallest level gap across the fan: {lo*1e3:+.2f} to "
              f"{hi*1e3:+.2f} meV")

    lv = [[r[0]] for r in res]          # (nB, 1) grid: dU is a single value
    lw = [[r[1]] for r in res]
    vv = np.full((len(B), 1, max(counts)), np.nan)
    for i, r in enumerate(res):
        vv[i, 0, :len(r[2])] = r[2]

    tag = "dU0" if DU == 0.0 else f"dU{round(DU*1e3)}"
    path = os.path.join(CACHE, f"ll_minimal_lowB_{tag}.mat")
    matio.save_levels(path, B, np.array([DU]), lv, lw, P.minimal(), MODEL,
                      extra={"valley": vv, "energy_window_eV": WINDOW,
                             "note": "Minimal six-band model (gamma0, gamma1 "
                                     "only), low field. Charge neutrality is "
                                     "exactly E = 0 for this model: delta and "
                                     "gamma4, which shift it in the full "
                                     "model, are switched off."})
    print(f"  wrote {path}")
