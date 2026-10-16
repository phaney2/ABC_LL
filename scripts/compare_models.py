"""Quantify the error of each approximation against the full six-band model,
and time the solver.  Run:  python3 scripts/compare_models.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import bands, landau  # noqa: E402
from abc_ll.params import DEFAULT as P  # noqa: E402

MEV = 1e3


def band_edges(du, model, kmax_ev=0.30, nk=1200, nth=241):
    """Conduction minimum / valence maximum over a full 2D k scan.

    Returns a dict with the edge energies, the |k| at which they occur (in
    units of hbar*v*k, eV), and the band energies at k=0.
    """
    k = np.linspace(1e-7, kmax_ev, nk) / P.hbar_v
    th = np.linspace(0.0, 2 * np.pi / 3, nth)  # C3 symmetric, one sector is enough
    kx = np.outer(k, np.cos(th))
    ky = np.outer(k, np.sin(th))
    lo, hi = bands.low_energy_pair(model, kx, ky, du, P)

    ic = np.unravel_index(np.argmin(hi), hi.shape)
    iv = np.unravel_index(np.argmax(lo), lo.shape)
    k0 = np.array([1e-9]) / P.hbar_v
    lo0, hi0 = bands.low_energy_pair(model, k0, 0 * k0, du, P)
    return {
        "ec": hi[ic], "kc": P.hbar_v * k[ic[0]],
        "ev": lo[iv], "kv": P.hbar_v * k[iv[0]],
        "ec0": hi0[0], "ev0": lo0[0],
    }


def zero_field_errors():
    print("=" * 78)
    print("ZERO FIELD: two-band error vs the full six-band model")
    print("  max |dE| over the low-energy pair, for hbar*v*k up to the listed cut")
    print("=" * 78)
    print(f"{'du (eV)':>8} {'cut (eV)':>9} {'minimal 2b':>13} {'extended 2b':>13} "
          f"{'6b(g0,g1) ':>12}")
    for du in (0.0, 0.05, 0.10, 0.20):
        for cut in (0.05, 0.12, 0.25):
            k = np.linspace(1e-6, cut, 400) / P.hbar_v
            # average over angle: gamma2+cubic makes the dispersion trigonal
            th = np.linspace(0, 2 * np.pi, 13)[:-1]
            kx = np.outer(k, np.cos(th))
            ky = np.outer(k, np.sin(th))
            ref = bands.low_energy_pair("six_band", kx, ky, du, P)
            row = []
            for m in ("two_band_minimal", "two_band", "six_band_minimal"):
                e = bands.low_energy_pair(m, kx, ky, du, P)
                row.append(max(np.max(np.abs(ref[i] - e[i])) for i in (0, 1)) * MEV)
            print(f"{du:8.2f} {cut:9.2f} {row[0]:10.2f} meV {row[1]:10.2f} meV "
                  f"{row[2]:9.2f} meV")
    print()


def gap_structure():
    print("=" * 78)
    print("ZERO FIELD: band-edge structure vs displacement field")
    print("  'hat' = how far the band edge lies below/above its k=0 value;")
    print("  a nonzero hat means a ring-shaped edge -> van Hove peak in the DOS")
    print("=" * 78)
    print(f"{'du':>6} | {'six-band: gap':>13} {'hat_c':>8} {'hat_v':>8} "
          f"{'hbar v k_c':>10} {'hbar v k_v':>10} | {'2b-min gap':>10} "
          f"{'2b-ext gap':>10}")
    for du in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20):
        s = band_edges(du, "six_band")
        m = band_edges(du, "two_band_minimal")
        x = band_edges(du, "two_band")
        print(f"{du:6.2f} | {(s['ec']-s['ev'])*MEV:10.2f}    "
              f"{(s['ec0']-s['ec'])*MEV:7.2f}  {(s['ev']-s['ev0'])*MEV:7.2f}  "
              f"{s['kc']:10.4f} {s['kv']:10.4f} | "
              f"{(m['ec']-m['ev'])*MEV:9.2f}  {(x['ec']-x['ev'])*MEV:9.2f}")
    print("  (energies in meV, hbar*v*k in eV)")
    print()


def _edge_levels(e, k):
    """(k lowest levels above E=0, k highest below), index-aligned.

    Comparing "the first k conduction Landau levels" is robust against the
    models putting different *numbers* of levels inside a fixed energy window.
    """
    up = np.sort(e[e > 0])[:k]
    dn = np.sort(e[e < 0])[::-1][:k]
    return up, dn


def _levels(model, B, du, window):
    return np.sort(np.concatenate([
        landau.solve_auto(model, B, du, P, valley=xi, window=window).energy
        for xi in (+1, -1)]))


def _edge_err(ref, test, k):
    r_up, r_dn = _edge_levels(ref, k)
    t_up, t_dn = _edge_levels(test, k)
    nu, nd = min(len(r_up), len(t_up)), min(len(r_dn), len(t_dn))
    d = np.concatenate([np.abs(r_up[:nu] - t_up[:nu]),
                        np.abs(r_dn[:nd] - t_dn[:nd])])
    return (np.max(d) if d.size else np.nan) * MEV


def landau_error_decomposition(window=0.3, k_edge=6):
    print("=" * 78)
    print("LANDAU LEVELS: where does the two-band error come from?")
    print(f"  max |dE| over the {k_edge} lowest conduction + {k_edge} highest")
    print("  valence levels, both valleys.  Columns isolate the three sources:")
    print("    (a) ladder substitution : two_band_minimal vs two_band_folded_minimal")
    print("    (b) two-band truncation : two_band_folded_minimal vs six_band_minimal")
    print("    (c) remote hoppings     : six_band_minimal        vs six_band")
    print("    total                   : two_band                vs six_band")
    print("  'E1' is the energy of the lowest conduction level in the six-band"
          " model.")
    print("=" * 78)
    print(f"{'B':>5} {'du':>6} {'E1':>8} | {'(a) ladder':>11} {'(b) 2-band':>11} "
          f"{'(c) remote':>11} | {'total':>9} {'hbar.om^2/g1':>12}")
    for B in (1.0, 2.0, 4.0, 8.0, 12.0):
        for du in (0.0, 0.10, 0.20):
            s = {m: _levels(m, B, du, window) for m in
                 ("two_band_minimal", "two_band", "two_band_folded_minimal",
                  "six_band_minimal", "six_band")}
            up = s["six_band"][s["six_band"] > 0]
            e1 = up[0] * MEV if up.size else np.nan
            a = _edge_err(s["two_band_folded_minimal"], s["two_band_minimal"], k_edge)
            b = _edge_err(s["six_band_minimal"], s["two_band_folded_minimal"], k_edge)
            c = _edge_err(s["six_band"], s["six_band_minimal"], k_edge)
            tot = _edge_err(s["six_band"], s["two_band"], k_edge)
            scale = float(P.omega(B)) ** 2 / P.gamma1 * MEV
            print(f"{B:5.1f} {du:6.2f} {e1:7.1f}  | {a:8.2f}    {b:8.2f}    "
                  f"{c:8.2f}    | {tot:7.2f}  {scale:10.1f}")
    print("  (all errors in meV)")
    print()


def timing():
    print("=" * 78)
    print("TIMING: one (B, du, valley) solve, window |E| < 0.3 eV")
    print("=" * 78)
    for model in landau.MODELS:
        for B in (1.0, 5.0, 15.0):
            n = landau.suggest_n_ll(model, B, P, window=0.3)
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                lv = landau.solve_levels(model, B, 0.1, P, n_ll=n, window=0.3)
            dt = (time.perf_counter() - t0) / reps
            dim = landau.NBANDS[model] * n
            print(f"  {model:18s} B={B:5.1f} T  n_ll={n:5d} (dim {dim:5d})  "
                  f"{dt*1e3:8.1f} ms  -> {len(lv):4d} levels")
    print()


if __name__ == "__main__":
    zero_field_errors()
    gap_structure()
    landau_errors()
    timing()
