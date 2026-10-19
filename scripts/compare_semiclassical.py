"""Score the semiclassical Onsager fan against the exact ABC Landau levels.

`abc_ll.landau` solves the same Hamiltonian exactly in a field, so unlike the
engine's usual setting there is ground truth here.  This script quantifies the
agreement so the semiclassics can be trusted (or not) feature by feature.

Two directions are reported, because they fail for different reasons:

``semi -> exact``  how far each semiclassical level sits from the nearest exact
    one.  This is the headline number: a semiclassical level with no exact
    partner is a spurious level.

``exact -> semi``  how far each exact level sits from the nearest semiclassical
    one.  This one is *expected* to be poor near charge neutrality -- the
    12-fold zero-mode multiplet (orbital n = 0,1,2 x spin x valley) is not a
    semiclassical object and no orbit-area argument can produce it.

Both are quoted in meV and as a fraction of the local exact level spacing,
which is the scale that matters for whether a fan is usable.

**Levels are counted out from charge neutrality by index, not by energy.**  The
lowest few are known to fit badly -- semiclassics is an expansion in 1/n and
the zero-mode multiplet is outside it -- so `N_SKIP` of them are dropped on
each side.  A fixed energy cut would not do: at 1 T the levels are ten times
denser than at 12 T, so the same cut in meV throws away ten times as many
indices and the score would drift with field for no physical reason.  The score
is also reported split into low and high index so the 1/n improvement is
visible rather than averaged away.

Run:  python3 scripts/compare_semiclassical.py six_band
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import landau  # noqa: E402
from abc_ll.dos import cnp_energy  # noqa: E402
from abc_ll.params import DEFAULT as _P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

_ALIAS = {"6band": "six_band", "6band_minimal": "six_band_minimal",
          "2band": "two_band_minimal", "2band_minimal": "two_band_minimal"}
_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "six_band")
MODEL = _ALIAS.get(_arg, _arg)
P = _P.minimal() if MODEL.endswith("minimal") else _P

#: which terms to score.  'S' is bare Onsager, 'SB' adds the Berry phase.
TERMS = os.environ.get("TERMS", "S,SB,SBM").split(",")
#: Landau levels to drop on each side of charge neutrality, counting outward.
#: The zero-mode multiplet is orbital n = 0,1,2, and semiclassics is an
#: expansion in 1/n, so these are expected to fit badly and say nothing about
#: whether the method works.
N_SKIP = int(os.environ.get("N_SKIP", 3))
#: boundary between the "low" and "high" index columns of the report
N_HIGH = int(os.environ.get("N_HIGH", 8))
#: subsample the grids; the exact solve is the expensive part here
DU_STRIDE = int(os.environ.get("DU_STRIDE", 5))
B_STRIDE = int(os.environ.get("B_STRIDE", 8))


def _nearest(a, b):
    """For each entry of `a`, the distance to the nearest entry of `b`."""
    if a.size == 0 or b.size == 0:
        return np.array([])
    return np.abs(a[:, None] - b[None, :]).min(axis=1)


def _index_from_cnp(levels, cnp):
    """Landau index of each level, counting outward from neutrality per side.

    Returns an integer array the same length as `levels`: 0 for the level
    nearest neutrality on its own side, 1 for the next, and so on.  Electron
    and hole sides are numbered independently.
    """
    idx = np.empty(levels.size, dtype=int)
    for side in (+1, -1):
        m = np.flatnonzero(np.sign(levels - cnp) == side)
        if m.size:
            idx[m[np.argsort(np.abs(levels[m] - cnp))]] = np.arange(m.size)
    return idx


def semiclassical_levels(S, keys, idu, ib, valley, term):
    """Union of the fan over pockets and Lifshitz segments, in meV."""
    out = []
    prefix = f"LL_{valley}_b{ib}_"
    for key in keys:
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):]            # e.g. "p0_SB" or "p1_SB_seg1"
        parts = tail.split("_", 1)
        if len(parts) != 2:
            continue
        suffix = parts[1]
        if suffix != term and not suffix.startswith(term + "_seg"):
            continue
        arr = S[key]                        # (nDU, nB, nmax+1)
        out.append(arr[idu])
    if not out:
        return np.zeros((0, 0))
    return np.concatenate(out, axis=-1)     # (nB, nlevels)


def main():
    path = os.path.join(CACHE, f"semiclassical_{MODEL}.mat")
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} not found.  Generate it with:\n"
            f"    python3 scripts/export_semiclassical.py {MODEL}")
    S = loadmat(path, squeeze_me=True)
    keys = [k for k in S if k.startswith("LL_")]

    du = np.atleast_1d(S["dU_eV"]).ravel()
    Blist = np.atleast_1d(S["Blist"]).ravel()
    e_win = float(S["energy_window_eV"]) * 1e3
    conv = np.atleast_2d(S["flux_converged"])

    idu_all = range(0, len(du), DU_STRIDE)
    ib_all = range(0, len(Blist), B_STRIDE)

    print(f"{MODEL}: semiclassical vs exact Landau levels")
    print(f"  window +-{e_win:.0f} meV, dropping the lowest {N_SKIP} levels "
          f"on each side of neutrality")
    print(f"  'low' is index {N_SKIP}-{N_HIGH-1}, 'high' is index "
          f"{N_HIGH} and up, both as a fraction of the exact level spacing")
    print()
    hdr = (f"{'dU':>6} {'B':>6} {'val':>4} {'band':>5} {'term':>4} {'nSemi':>6} "
           f"{'nExact':>7} {'spacing':>8} {'semi>ex':>9} {'ex>semi':>9} "
           f"{'low/sp':>7} {'high/sp':>8}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for idu in idu_all:
        cnp = cnp_energy(float(du[idu]), P, model=MODEL) * 1e3
        gapless = not bool(np.all(conv[idu]))
        for ib in ib_all:
            B = float(Blist[ib])
            for valley, xi in (("K", +1), ("Kp", -1)):
                lev = landau.solve_auto(MODEL, B, float(du[idu]), P,
                                        valley=xi, window=e_win * 1e-3 * 1.2)
                ex_all = np.sort(lev.energy * 1e3)
                ex_all = ex_all[np.abs(ex_all) < e_win]
                if ex_all.size < 2 * N_SKIP + 4:
                    continue
                nidx = _index_from_cnp(ex_all, cnp)
                keep = nidx >= N_SKIP
                ex, ex_n = ex_all[keep], nidx[keep]
                if ex.size < 3:
                    continue
                spacing = float(np.median(np.diff(np.sort(ex))))
                # the energy below which levels were dropped, per side, so the
                # same cut can be applied to the semiclassical set
                hi_cut = ex[ex > cnp].min() if np.any(ex > cnp) else cnp
                lo_cut = ex[ex < cnp].max() if np.any(ex < cnp) else cnp

                # Score the two bands separately.  Merging them lets an exact
                # conduction level be matched by a valence semiclassical one
                # near the gap, which flatters the result; and the conduction
                # and valence sides turn out to behave differently, which is
                # exactly what a merged number would hide.
                for band, bname in ((1, 'cond'), (0, 'val')):
                  for term in TERMS:
                    a = semiclassical_levels(S, keys, idu, band, valley, term)
                    if not a.size:
                        continue
                    sm = a[ib]
                    sm = sm[np.isfinite(sm)]
                    side = (sm >= hi_cut) if band == 1 else (sm <= lo_cut)
                    sm = sm[(np.abs(sm) < e_win) & side]
                    exs = ex[ex >= hi_cut] if band == 1 else ex[ex <= lo_cut]
                    exn = ex_n[ex >= hi_cut] if band == 1 else ex_n[ex <= lo_cut]
                    if sm.size == 0 or exs.size < 3:
                        continue

                    d_se = _nearest(sm, exs)
                    d_es = _nearest(exs, sm)

                    # split the exact->semi error by Landau index: semiclassics
                    # is a 1/n expansion, so this should fall with index and a
                    # single average would hide that
                    lo_m = (exn >= N_SKIP) & (exn < N_HIGH)
                    hi_m = exn >= N_HIGH
                    med_lo = float(np.median(d_es[lo_m])) if lo_m.any() else np.nan
                    med_hi = float(np.median(d_es[hi_m])) if hi_m.any() else np.nan

                    row = dict(du=du[idu] * 1e3, B=B, valley=valley,
                               band=bname, term=term,
                               n_semi=sm.size, n_exact=exs.size,
                               spacing=spacing,
                               med_se=float(np.median(d_se)),
                               med_es=float(np.median(d_es)),
                               med_lo=med_lo, med_hi=med_hi,
                               gapless=gapless)
                    rows.append(row)
                    flag = " *" if (gapless and term != "S") else ""
                    print(f"{row['du']:6.0f} {B:6.1f} {valley:>4} {bname:>5} "
                          f"{term:>4} "
                          f"{sm.size:6d} {exs.size:7d} {spacing:8.2f} "
                          f"{row['med_se']:9.2f} {row['med_es']:9.2f} "
                          f"{med_lo/spacing:7.3f} {med_hi/spacing:8.3f}{flag}")

    print()
    print("  semi>ex  median distance from each semiclassical level to the "
          "nearest exact one (meV)")
    print("  ex>semi  the reverse (meV)")
    print(f"  low/sp   ex>semi over Landau index {N_SKIP}-{N_HIGH-1}, as a "
          "fraction of the level spacing")
    print(f"  high/sp  the same for index {N_HIGH} and up -- the number that "
          "says whether the fan is usable")
    print("  *        gapless at this dU, so the Berry term is absent and SB "
          "is really S")

    if rows:
        out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        out["band"] = np.array([r["band"] for r in rows], dtype=object)
        out["model"] = MODEL
        out["readme"] = (
            "Semiclassical Onsager fan scored against the exact ABC Landau "
            "levels from abc_ll.landau, same Hamiltonian and parameters.  "
            "med_se is the median distance from a semiclassical level to the "
            "nearest exact one; med_es is the reverse and is expected to be "
            "large near charge neutrality, where the exact spectrum has a "
            "zero-mode multiplet that no semiclassical orbit can reproduce.")
        p = os.path.join(CACHE, f"semiclassical_vs_exact_{MODEL}.mat")
        savemat(p, out, do_compression=True)
        print(f"\n  wrote {p}")

        print(f"\n  {'term':>5} {'low index':>11} {'high index':>12}   "
              f"(median over gapped dU, in level spacings)")
        for bname in ('cond', 'val'):
            for term in TERMS:
                sel = [r for r in rows if r["term"] == term
                       and r["band"] == bname and not r["gapless"]]
                if sel:
                    lo = np.nanmedian([r["med_lo"] / r["spacing"] for r in sel])
                    hi = np.nanmedian([r["med_hi"] / r["spacing"] for r in sel])
                    print(f"  {bname:>5} {term:>5} {lo:9.3f} {hi:12.3f}")


if __name__ == "__main__":
    main()
