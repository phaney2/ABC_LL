"""MATLAB (.mat) export.

Everything written here is self-describing: each file carries a ``units``
struct and a ``readme`` string, plus the full parameter set that produced it,
so a .mat can be interpreted without the Python source.

Array orientation follows the natural Python shape and `scipy.io.savemat`
preserves it, so a ``(nB, nU)`` array reads as ``nB x nU`` in MATLAB.
"""

from __future__ import annotations

import numpy as np
from scipy.io import savemat

from .params import ABCParams, NPHI_1T


def _cnp(du, p):
    """Charge-neutrality energy for each dU, same shape as ``du``.

    Imported lazily to keep this module free of a hard dependency on the
    DOS machinery.  Neutrality is **not** at E = 0 -- see `dos.cnp_energy`
    -- so every export carries it, and the MATLAB side uses it as the
    reference for filling.
    """
    from .dos import cnp_energy

    du = np.asarray(du, float)
    flat = np.array([cnp_energy(float(x), p) for x in du.ravel()])
    return flat.reshape(du.shape)


CNP_NOTE = (" Charge neutrality is NOT at E = 0: delta and gamma4 shift it to "
            "about +6.3 meV at dU = 0. cnp_eV holds it, with the same shape "
            "as the dU grid. Measure filling from cnp_eV, not from zero.")


def param_struct(p: ABCParams) -> dict:
    """The tight-binding parameter set, as a MATLAB struct."""
    return {
        "gamma0": p.gamma0, "gamma1": p.gamma1, "gamma2": p.gamma2,
        "gamma3": p.gamma3, "gamma4": p.gamma4, "delta": p.delta,
        "a_m": p.a, "d0_m": p.d0,
        "hbar_v_eV_m": p.hbar_v, "v_ms": p.v_ms,
        "cell_area_m2": p.cell_area,
        "note": ("ABC trilayer SWMc parameters, eV. Basis (A1,B1,A2,B2,A3,B3); "
                 "dimers (B1,A2) and (B2,A3); gamma2 couples the vertically "
                 "aligned A1 and B3 and enters as gamma2/2."),
    }


def gate_struct(gate) -> dict:
    return {
        "kappa_m2_per_eV": gate.kappa,
        "du_max_eV": gate.du_max,
        "n_max_m2": gate.n_max,
        "n_max_cm2": gate.n_max / 1e4,
        "label": gate.label,
        "note": ("Single back gate: n = kappa * dU, so density and "
                 "displacement field are locked. dU < 0 is the hole side."),
    }


def broadening_struct(br) -> dict:
    return {
        "gamma0_eV": br.gamma0,
        "scaling": br.scaling,
        "lineshape": br.lineshape,
        "note": ("gamma(B) = gamma0*sqrt(B) for scaling='sqrtB'. gamma0 is "
                 "the HWHM for a Lorentzian, the standard deviation for a "
                 "Gaussian. Gaussian is preferred: Lorentzian 1/E^2 tails "
                 "make the in-gap DOS depend on the retained energy range."),
    }


def save_dos_map(path, B, du, mu, dos, dos_layer, p, gate, broadening, model):
    """Write the (B, dU) DOS map.

    Shapes: ``B`` is (nB,), ``du`` is (nU,), ``mu``/``dos`` are (nB, nU),
    ``dos_layer`` is (nB, nU, 3) ordered bottom, middle, top.
    """
    B = np.asarray(B, float)
    du = np.asarray(du, float)
    dos_layer = np.asarray(dos_layer, float)
    from .dos import moire_cell_area

    nphi = NPHI_1T * B[:, None]                     # (nB, 1)
    nu = gate.density(du)[None, :] / nphi           # (nB, nU)
    # Two different "fillings". nu is the Landau filling factor and depends
    # on B; filling_moire counts states per moire cell, is linear in gate
    # voltage and is B-independent -- it is the natural x-axis for a gate
    # sweep, running -4 to +4 over the full dU range.
    filling_moire = gate.density(du) * moire_cell_area()
    total = dos_layer.sum(axis=-1)

    cnp = _cnp(du, p)
    out = {
        "model": model,
        "B_T": B,
        "dU_eV": du,
        "dU_meV": du * 1e3,
        "cnp_eV": cnp,
        "cnp_meV": cnp * 1e3,
        "n_m2": gate.density(du),
        "n_cm2": gate.density(du) / 1e4,
        "nu": nu,
        "filling_moire": filling_moire,
        "nphi_m2": NPHI_1T * B,
        "mu_eV": mu,
        "mu_meV": np.asarray(mu) * 1e3,
        "dos_m2_eV": dos,
        "dos_cm2_eV": np.asarray(dos) / 1e4,
        "dos_layer_m2_eV": dos_layer,
        "dos_layer_cm2_eV": dos_layer / 1e4,
        "top_fraction": np.divide(dos_layer[..., 2], total,
                                  out=np.full_like(total, np.nan),
                                  where=total > 0),
        "params": param_struct(p),
        "gate": gate_struct(gate),
        "broadening": broadening_struct(broadening),
        "units": {
            "B_T": "tesla",
            "dU_eV": "eV, total interlayer potential difference u3 - u1",
            "n_m2": "m^-2, carrier density relative to charge neutrality",
            "nu": "Landau filling factor n*h/(e*B), all four flavours; (nB x nU)",
            "filling_moire": "states per moire cell, linear in gate voltage "
                             "and independent of B; (nU,)",
            "mu_eV": "eV, chemical potential, absolute (neutrality = cnp_eV)",
            "cnp_eV": "eV, charge-neutrality energy at each dU -- not zero",
            "dos_m2_eV": "states m^-2 eV^-1, both valleys, spin included",
            "dos_layer_m2_eV": "same, resolved by layer [bottom, middle, top]",
            "top_fraction": "dos_layer(:,:,3) ./ dos, dimensionless",
        },
        "readme": (
            "ABC (rhombohedral) trilayer graphene Landau-level DOS at the "
            "Fermi level, over magnetic field B and interlayer potential "
            "difference dU. Single back gate, so n = kappa*dU is locked to "
            "dU and does not change with B; the chemical potential moves "
            "instead. Lines of constant nu are therefore straight lines "
            "through the origin of the (B, dU) plane. Layers are ordered "
            "[bottom, middle, top]; the top layer sits at u3 = +dU/2. "
            "Index order is (B, dU) and (B, dU, layer). Single-particle, "
            "no interactions, no moire band folding." + CNP_NOTE),
    }
    savemat(str(path), out, do_compression=True)
    return path


def save_levels_nu(path, B, nu, du2d, levels, layer_weight, valley, p, gate,
                   model, window, note=""):
    """Write levels on a (B, filling) grid, where dU varies with B.

    For a single gate ``nu = kappa*dU / (eB/h)``, so a rectangular grid in
    (B, nu) corresponds to ``dU = nu * (eB/h) / kappa``, which depends on B.
    ``du2d`` is therefore ``(nB, nNu)`` rather than a 1-D axis.
    """
    B = np.asarray(B, float)
    nu = np.asarray(nu, float)
    nB, nN = len(B), len(nu)
    counts = np.array([[len(levels[i][j]) for j in range(nN)]
                       for i in range(nB)], dtype=int)
    nmax = int(counts.max()) if counts.size else 0

    E = np.full((nB, nN, nmax), np.nan)
    W = np.full((nB, nN, nmax, 3), np.nan)
    V = np.full((nB, nN, nmax), np.nan)
    for i in range(nB):
        for j in range(nN):
            k = counts[i, j]
            E[i, j, :k] = levels[i][j]
            W[i, j, :k, :] = layer_weight[i][j]
            V[i, j, :k] = valley[i][j]

    cnp = _cnp(du2d, p)
    out = {
        "model": model,
        "B_T": B,
        "nu": nu,
        "dU_eV": du2d,
        "dU_meV": np.asarray(du2d) * 1e3,
        "cnp_eV": cnp,
        "cnp_meV": cnp * 1e3,
        "nphi_m2": NPHI_1T * B,
        "energy_eV": E,
        "energy_meV": E * 1e3,
        "layer_weight": W,
        "valley": V,
        "n_levels": counts,
        "energy_window_eV": window,
        "params": param_struct(p),
        "gate": gate_struct(gate),
        "units": {
            "B_T": "tesla",
            "nu": "Landau filling factor, all four flavours",
            "dU_eV": "eV, (nB x nNu): the dU that gives this nu at this B",
            "energy_eV": "eV, absolute (neutrality = cnp_eV, not zero)",
            "cnp_eV": "eV, charge-neutrality energy, (nB x nNu)",
            "n_levels": "valid entries along dim 3 at each (B, nu)",
        },
        "readme": (
            "Landau levels of ABC trilayer graphene on a (B, filling) grid. "
            "Single back gate, so dU is set by nu and B via "
            "dU = nu*(e*B/h)/kappa and varies across the grid -- see dU_eV. "
            "Both valleys merged and sorted by energy, NaN-padded along dim 3 "
            "to the maximum level count; use n_levels. layer_weight is the "
            "fraction on [bottom, middle, top] and valley is +1 for K, -1 for "
            "K'. No broadening is applied here: build the DOS in MATLAB so "
            "the linewidth stays a free parameter." + CNP_NOTE + " " + note),
    }
    savemat(str(path), out, do_compression=True)
    return path


def save_levels(path, B, du, levels, layer_weight, p, model, extra=None):
    """Write raw Landau levels on a (B, dU) grid, NaN-padded.

    ``levels`` is a nested list indexed ``[iB][iU]`` of 1-D energy arrays
    (any lengths); ``layer_weight`` the matching ``(n_lev, 3)`` arrays, or
    ``None``.  They are packed into ``(nB, nU, n_max)`` and
    ``(nB, nU, n_max, 3)`` arrays padded with NaN, with ``n_levels``
    recording the true count at each point.
    """
    B = np.atleast_1d(np.asarray(B, float))
    du = np.atleast_1d(np.asarray(du, float))
    nB, nU = len(B), len(du)
    counts = np.array([[len(levels[i][j]) for j in range(nU)]
                       for i in range(nB)], dtype=int)
    nmax = int(counts.max()) if counts.size else 0

    E = np.full((nB, nU, nmax), np.nan)
    W = np.full((nB, nU, nmax, 3), np.nan) if layer_weight is not None else None
    for i in range(nB):
        for j in range(nU):
            k = counts[i, j]
            E[i, j, :k] = levels[i][j]
            if W is not None:
                W[i, j, :k, :] = layer_weight[i][j]

    cnp = _cnp(du, p)
    out = {
        "model": model,
        "B_T": B,
        "dU_eV": du,
        "dU_meV": du * 1e3,
        "cnp_eV": cnp,
        "cnp_meV": cnp * 1e3,
        "energy_eV": E,
        "energy_meV": E * 1e3,
        "n_levels": counts,
        "params": param_struct(p),
        "units": {
            "energy_eV": "eV, absolute (neutrality = cnp_eV, not zero)",
            "cnp_eV": "eV, charge-neutrality energy at each dU -- not zero",
            "n_levels": "number of valid entries along dim 3 at each (B, dU)",
        },
        "readme": (
            "Landau levels of ABC trilayer graphene on a (B, dU) grid, both "
            "valleys merged and sorted by energy, NaN-padded along the third "
            "dimension to the maximum level count. Use n_levels to find the "
            "valid entries. layer_weight, if present, is the fraction of "
            "each level on [bottom, middle, top]; rows sum to 1." + CNP_NOTE),
    }
    if W is not None:
        out["layer_weight"] = W
    if extra:
        out.update(extra)
    savemat(str(path), out, do_compression=True)
    return path
