"""Zero-field band structure and density of states, per interlayer potential.

Companion to levels_<model>.mat: same dU grid, so a slider index selects the
same physical point in both files.

Two products per dU:
  bands_eV   E(k) along a cut through K at ky = 0, kx from -kmax to +kmax.
             Both signs of kx are kept because trigonal warping makes the
             two directions inequivalent.
  dos_m2_eV  the zero-field DOS, on the same energy grid the Landau-level
             plots use, so the two panels can be read against each other.

The DOS is a 2D k integral. Two symmetries cut the cost: the model is C3
about K, so one 120-degree sector times three suffices; and time reversal
maps K to K' at the same energy (dU is a scalar potential), so one valley
doubled gives the total.

Run:  python3 scripts/export_zerofield.py six_band
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import bands as bandmod, dos as dosmod, matio  # noqa: E402
from abc_ll.params import DEFAULT as _P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

_ALIAS = {"6band": "six_band", "6band_minimal": "six_band_minimal",
          "2band": "two_band", "2band_minimal": "two_band_minimal"}
_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "six_band")
MODEL = _ALIAS.get(_arg, _arg)
P = _P.minimal() if MODEL.endswith("minimal") else _P

NDU = int(os.environ.get("NDU", 41))
DU_MAX = float(os.environ.get("DU_MAX", 0.20))
E_WIN = float(os.environ.get("E_WIN", 0.16))      # eV, matches the LL window
NE = int(os.environ.get("NE", 1281))              # 0.25 meV bins over +-160
KMAX_EV = float(os.environ.get("KMAX_EV", 0.40))  # cut extent, as hbar*v*k
NK_CUT = int(os.environ.get("NK_CUT", 801))
NK_DOS = int(os.environ.get("NK_DOS", 900))       # radial samples
NTH_DOS = int(os.environ.get("NTH_DOS", 300))     # over one C3 sector

G_SPIN = 2


def band_cut(du):
    """E(kx) at ky = 0, kx from -kmax to +kmax, valley K."""
    kx = np.linspace(-KMAX_EV, KMAX_EV, NK_CUT) / P.hbar_v
    return bandmod.bands(MODEL, kx, np.zeros_like(kx), du, P, valley=+1)


def zero_field_dos(du, edges):
    """Binned zero-field DOS [states m^-2 eV^-1] on the given energy edges."""
    k = np.linspace(1e-9, KMAX_EV, NK_DOS) / P.hbar_v
    th = np.linspace(0.0, 2 * np.pi / 3, NTH_DOS, endpoint=False)
    dk = k[1] - k[0]
    dth = th[1] - th[0]
    kx = np.outer(k, np.cos(th))
    ky = np.outer(k, np.sin(th))
    e = bandmod.bands(MODEL, kx, ky, du, P, valley=+1)      # (nk, nth, nb)

    # area element k dk dth, shared by every band at that k point;
    # x3 for the other C3 sectors, x2 for the other valley (time reversal),
    # x G_SPIN for spin, and /(2 pi)^2 for the k-space density of states
    w = (np.outer(k, np.ones_like(th)) * dk * dth)[..., None]
    w = np.broadcast_to(w, e.shape)
    scale = 3.0 * 2.0 * G_SPIN / (2 * np.pi) ** 2

    hist, _ = np.histogram(e.ravel(), bins=edges, weights=w.ravel())
    return scale * hist / np.diff(edges)


if __name__ == "__main__":
    du = np.linspace(-DU_MAX, DU_MAX, NDU)
    edges = np.linspace(-E_WIN, E_WIN, NE + 1)
    ecen = 0.5 * (edges[:-1] + edges[1:])
    kx_ev = np.linspace(-KMAX_EV, KMAX_EV, NK_CUT)

    print(f"{MODEL}: zero-field bands and DOS at {NDU} dU values")
    t0 = time.perf_counter()
    nb = 2 if MODEL.startswith("two_band") else 6
    bands_out = np.empty((NDU, NK_CUT, nb))
    dos_out = np.empty((NDU, NE))
    cnp = np.empty(NDU)
    for i, u in enumerate(du):
        bands_out[i] = band_cut(u)
        dos_out[i] = zero_field_dos(u, edges)
        cnp[i] = dosmod.cnp_energy(float(u), P, model=MODEL)
    print(f"  {time.perf_counter()-t0:.0f} s")

    gate = dosmod.moire_gate()
    out = {
        "model": MODEL,
        "dU_eV": du, "dU_meV": du * 1e3,
        "cnp_eV": cnp, "cnp_meV": cnp * 1e3,
        "hbarv_kx_eV": kx_ev,
        "bands_eV": bands_out, "bands_meV": bands_out * 1e3,
        "E_eV": ecen, "E_meV": ecen * 1e3,
        "dos_m2_eV": dos_out, "dos_cm2_eV": dos_out / 1e4,
        "n_m2": gate.density(du), "n_cm2": gate.density(du) / 1e4,
        "filling_moire": 4.0 * du / gate.du_max,
        "params": matio.param_struct(P),
        "gate": matio.gate_struct(gate),
        "units": {
            "hbarv_kx_eV": "hbar*v*kx in eV; cut through K at ky = 0",
            "bands_eV": "(nDU, nk, nbands) zero-field bands, valley K",
            "dos_m2_eV": "(nDU, nE) zero-field DOS, states m^-2 eV^-1, "
                         "both valleys and spin included",
            "cnp_eV": "charge-neutrality energy -- not zero for the full model",
        },
        "readme": (
            "Zero-field band structure and DOS for ABC trilayer graphene, on "
            "the same dU grid as the matching levels_<model>.mat, so the same "
            "index selects the same physical point. Bands are a cut at "
            "ky = 0 with kx of both signs, since trigonal warping makes the "
            "two directions inequivalent. The DOS is a full 2D k integral, "
            "using C3 symmetry and time reversal to cut the cost, in the "
            "same units as the Landau-level DOS so the two are comparable."),
    }
    path = os.path.join(CACHE, f"zerofield_{MODEL}.mat")
    savemat(path, out, do_compression=True)
    print(f"  wrote {path}")
