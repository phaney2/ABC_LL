"""Produce the two .mat files the MATLAB script needs, for one model.

One uniform layout, so matlab/plot_abc.m has a single thing to read:

    levels_<model>.mat   raw Landau levels on a (B, dU) grid -> 'Edos'
    dosmap_<model>.mat   DOS at E_F on a (B, dU) grid        -> 'ndos'

The levels file carries no broadening: the MATLAB side builds the DOS from
the levels so the linewidth stays a free parameter.  The DOS map is
necessarily pre-broadened, because finding E_F needs a linewidth.

Run:  python3 scripts/export_for_matlab.py six_band

Accepts the model as a plain argument (works in cmd.exe, PowerShell and
bash alike), or as the MODEL environment variable. Short names 6band,
6band_minimal and 2band are accepted too, matching matlab/plot_abc.m.

Other settings come from the environment: BMIN, BMAX, NB, NDU, NB_MAP,
NDU_MAP, WINDOW, SKIP_LEVELS, SKIP_MAP.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import dos, landau, matio  # noqa: E402
from abc_ll.params import DEFAULT as _P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

_ALIAS = {"6band": "six_band", "6band_minimal": "six_band_minimal",
          "2band": "two_band", "2band_minimal": "two_band_minimal"}
_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "six_band")
_arg = _arg.split("=", 1)[1] if _arg.startswith("MODEL=") else _arg
MODEL = _ALIAS.get(_arg, _arg)
if MODEL not in ("six_band", "six_band_minimal", "two_band",
                 "two_band_minimal"):
    raise SystemExit(f"unknown model {_arg!r}; expected one of "
                     f"six_band, six_band_minimal, two_band")
P = _P.minimal() if MODEL.endswith("minimal") else _P
WINDOW = float(os.environ.get("WINDOW", 0.16))
BMIN = float(os.environ.get("BMIN", 1.0))
BMAX = float(os.environ.get("BMAX", 12.0))
#: Levels grid: B is the plot axis so wants to be dense; dU is only a
#: selector, so a coarse set of values is plenty.
NB, NDU = int(os.environ.get("NB", 160)), int(os.environ.get("NDU", 21))
#: DOS map: both axes are plotted, so both need resolution.
NB_MAP = int(os.environ.get("NB_MAP", 100))
NDU_MAP = int(os.environ.get("NDU_MAP", 241))

GATE = dos.moire_gate()
#: Broadening for the DOS map only.  Finding E_F needs a linewidth, so
#: unlike the levels file this one is baked in -- GAMMA0 in plot_abc.m has
#: no effect on 'ndos'.  Sharpen the map by lowering this and re-running.
GAMMA0 = float(os.environ.get("GAMMA0", 1.0e-3))
BROAD = dos.Broadening(gamma0=GAMMA0,
                       scaling=os.environ.get("SCALING", "sqrtB"),
                       lineshape=os.environ.get("LINESHAPE", "gaussian"))


def _levels(args):
    B, du = args
    es, ws, vs = [], [], []
    for xi in (+1, -1):
        lv = landau.solve_auto(MODEL, B, du, P, valley=xi, window=WINDOW,
                               layers=True)
        es.append(lv.energy)
        ws.append(lv.layer_weight)
        vs.append(np.full(len(lv), xi))
    e = np.concatenate(es)
    o = np.argsort(e)
    return e[o], np.concatenate(ws, axis=0)[o], np.concatenate(vs)[o]


def _dospoint(args):
    B, du = args
    try:
        mu, d, dl, _ = dos.fermi_level_and_dos(MODEL, B, du, P, GATE, BROAD,
                                               layers=True)
    except RuntimeError:
        return (np.nan,) * 5
    return (mu, d, dl[0], dl[1], dl[2])


SKIP_LEVELS = os.environ.get("SKIP_LEVELS", "") not in ("", "0")
SKIP_MAP = os.environ.get("SKIP_MAP", "") not in ("", "0")


def main():
    os.makedirs(CACHE, exist_ok=True)

    if not SKIP_LEVELS:
        _levels_file()
    if not SKIP_MAP:
        _map_file()


def _levels_file():
    B = np.linspace(BMIN, BMAX, NB)
    du = np.linspace(-0.20, 0.20, NDU)
    print(f"{MODEL}: levels on {NB} x {NDU}, |E| < {WINDOW*1e3:.0f} meV")
    t0 = time.perf_counter()
    pts = [(b, d) for b in B for d in du]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_levels, pts, chunksize=4))
    lv = [[res[i * NDU + j][0] for j in range(NDU)] for i in range(NB)]
    lw = [[res[i * NDU + j][1] for j in range(NDU)] for i in range(NB)]
    nmax = max(len(x) for r in lv for x in r)
    vv = np.full((NB, NDU, nmax), np.nan)
    for i in range(NB):
        for j in range(NDU):
            v = res[i * NDU + j][2]
            vv[i, j, :len(v)] = v
    path = os.path.join(CACHE, f"levels_{MODEL}.mat")
    matio.save_levels(path, B, du, lv, lw, P, MODEL, gate=GATE,
                      extra={"valley": vv, "energy_window_eV": WINDOW})
    print(f"  {time.perf_counter()-t0:.0f} s, up to {nmax} levels -> {path}")


def _map_file():
    Bm = np.linspace(BMIN, BMAX, NB_MAP)
    dum = np.linspace(-0.20, 0.20, NDU_MAP)
    print(f"{MODEL}: DOS map on {NB_MAP} x {NDU_MAP}, "
          f"gamma0 = {GAMMA0*1e3:.3f} meV, {BROAD.lineshape}")
    t0 = time.perf_counter()
    pts = [(b, d) for b in Bm for d in dum]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_dospoint, pts, chunksize=8))
    r = np.array(res).reshape(NB_MAP, NDU_MAP, 5)
    bad = int(np.sum(~np.isfinite(r[..., 1])))
    # Tag the filename with the linewidth: the map is pre-broadened, so a
    # file is only meaningful together with the gamma it was built at, and
    # two gammas must be able to coexist.
    path = os.path.join(CACHE, f"dosmap_{MODEL}_g{GAMMA0*1e3:g}meV.mat")
    matio.save_dos_map(path, Bm, dum, r[..., 0], r[..., 1], r[..., 2:],
                       P, GATE, BROAD, MODEL)
    print(f"  {time.perf_counter()-t0:.0f} s, {bad} failed -> {path}")


if __name__ == "__main__":
    main()
