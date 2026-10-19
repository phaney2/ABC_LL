"""ABC trilayer band structure in the form the semiclassical engine consumes.

The engine lives in a separate tree (`hofstadter/source_code/semiclassical`),
where it is driven by a moire bilayer band structure.  Nothing in it is specific
to that system: `isoenergy.get_energy_resolved_data` and
`onsager.onsager_fan_band` take plain per-band arrays and have no internal
imports.  This module produces those arrays for ABC trilayer graphene so the
same orbit-area / Onsager-quantization machinery can be pointed at it.

Three conventions have to match the engine exactly or the result is silently
wrong rather than visibly broken:

**Flattening.**  `isoenergy.py` reshapes with ``E.reshape(nk2, nk1, order='F')``
and linearizes enclosed points as ``n1*nk2 + n2``.  So the flat index is
``n1*nk2 + n2``: the n2 (y) index varies fastest.  :func:`kmesh` builds that.

**vol_M.**  The engine only ever uses it through
``cell_area = (2*pi)**2 / (vol_M*nk1*nk2)``, the k-space area per mesh cell.
There is no moire cell here, so `vol_M` is just the number that makes
`cell_area` come out right -- see :func:`vol_M_for`.  A consequence is that the
mesh **must be isotropic**: `_shoelace_area` works in grid-index coordinates and
multiplies by a single `cell_area`, so unequal dkx and dky would scale every
orbit area by dky/dkx without any warning.

**Units.**  E in meV, Oz in m^2, Lz in meV*m^2, kpoints in Angstrom^-1, vol_M in
m^2.  This package works in eV and metres, so the conversion happens on the way
out, in :func:`band_structure`.

The patch is a finite piece of continuum k-space, not a torus, so downstream
calls must pass ``periodic=False``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from . import bands as bandmod
from .params import ABCParams

#: Where the semiclassical engine lives.  Overridable so the path is not baked
#: into the repository; only `isoenergy.py` and `onsager.py` are imported, and
#: neither has any internal dependency.
ENGINE_DIR = os.environ.get(
    "ABC_SEMICLASSICAL_DIR",
    r"C:\Users\phaney\wrk\hofstadter\source_code\semiclassical")

#: Landau-level counting: the engine is per-valley and spinless, so a level
#: holds e*B/h states and the spin factor is applied only when a DOS is built.
G_SPIN = 2


def engine():
    """Import the engine's two post-processing modules.

    Returns ``(get_energy_resolved_data, onsager_fan_band, isoenergy_areas)``.
    """
    if not os.path.isdir(ENGINE_DIR):
        raise RuntimeError(
            f"semiclassical engine not found at {ENGINE_DIR!r}.  Set "
            "ABC_SEMICLASSICAL_DIR to the directory holding isoenergy.py and "
            "onsager.py (hofstadter/source_code/semiclassical).")
    if ENGINE_DIR not in sys.path:
        sys.path.insert(0, ENGINE_DIR)
    from isoenergy import get_energy_resolved_data, isoenergy_areas
    from onsager import onsager_fan_band

    return get_energy_resolved_data, onsager_fan_band, isoenergy_areas


# --------------------------------------------------------------------- mesh

def kmesh(kmax, nk):
    """Isotropic square patch of k-space centred on the valley point.

    ``kmax`` is in m^-1 and the patch runs ``[-kmax, kmax]`` on both axes, so
    the spacing is the same in x and y -- required, see the module docstring.

    Returns ``(kx, ky, dk)`` with kx, ky flattened in the engine's order:
    ``index = n1*nk2 + n2``, n2 (the y index) varying fastest.
    """
    ax = np.linspace(-kmax, kmax, nk)
    dk = ax[1] - ax[0]
    # meshgrid with x slow and y fast, then C-order ravel, gives n1*nk + n2
    KX, KY = np.meshgrid(ax, ax, indexing="ij")
    return KX.ravel(), KY.ravel(), dk


def vol_M_for(nk1, nk2, dk):
    """The `vol_M` that makes the engine's `cell_area` equal to ``dk**2``.

    `isoenergy.py` computes ``cell_area = (2*pi)**2/(vol_M*nk1*nk2)``; inverting
    that for ``cell_area = dk**2`` gives this.  Units: dk in m^-1 -> vol_M in
    m^2, which is what the engine documents.
    """
    return (2.0 * np.pi) ** 2 / (nk1 * nk2 * dk**2)


# ------------------------------------------------------- velocity operators

_H_BUILDER = {
    "two_band_minimal": lambda kx, ky, du, p, xi: bandmod.h_two_band_k(
        kx, ky, du, p, xi, True),
    "two_band": lambda kx, ky, du, p, xi: bandmod.h_two_band_k(
        kx, ky, du, p, xi, False),
    "six_band_minimal": lambda kx, ky, du, p, xi: bandmod.h_six_band_k(
        kx, ky, du, p, xi, True),
    "six_band": lambda kx, ky, du, p, xi: bandmod.h_six_band_k(
        kx, ky, du, p, xi, False),
}


def hamiltonian(model, kx, ky, du, p: ABCParams, valley=1):
    """The zero-field Hamiltonian of `model`, shape ``(..., nb, nb)`` in eV."""
    try:
        build = _H_BUILDER[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; choose from {sorted(_H_BUILDER)}") from None
    return build(kx, ky, du, p, valley)


def dH_dk(model, kx, ky, du, p: ABCParams, valley=1, h=None):
    """``(dH/dkx, dH/dky)`` in eV*m, by a 5-point central stencil.

    The stencil
    ``(f(-2h) - 8 f(-h) + 8 f(h) - f(2h)) / (12 h)``
    is algebraically exact for polynomials up to degree 4.  Every entry of the
    six-band Hamiltonian is linear in ``v*pi`` and every entry of the two-band
    one is at most cubic, so this is exact for both models and there is no
    hand-derived derivative to get wrong -- only roundoff, ~eps*|H|/h.

    ``h`` defaults to a small fraction of the inverse Compton-like scale
    ``1/hbar_v``, which keeps it well-conditioned for any patch size.
    """
    if h is None:
        h = 1e-4 / p.hbar_v          # m^-1; gives h*hbar_v = 1e-4 eV

    def f(dx, dy):
        return hamiltonian(model, kx + dx, ky + dy, du, p, valley)

    dx = (f(-2 * h, 0) - 8 * f(-h, 0) + 8 * f(h, 0) - f(2 * h, 0)) / (12 * h)
    dy = (f(0, -2 * h) - 8 * f(0, -h) + 8 * f(0, h) - f(0, 2 * h)) / (12 * h)
    return dx, dy


# ------------------------------------------------------- Kubo Oz and Lz

#: k-points per chunk in :func:`kubo`.  The Kubo sum holds a dozen
#: ``(chunk, nb, nb)`` arrays at once, so an unchunked 376x376 six-band mesh
#: peaks near half a gigabyte -- times however many workers are running.
KUBO_CHUNK = int(os.environ.get("ABC_KUBO_CHUNK", 20000))


def kubo(model, kx, ky, du, p: ABCParams, valley=1, eta=2e-3):
    """Eigenvalues, Berry curvature and orbital moment on a k grid.

    Mirrors `bandstructure.py::_kpoint_worker` in the engine term for term, so
    the signs that its Onsager validation established carry over:

        den   = E_n - E_m
        prod  = Im[vx(n,m) * vy(m,n)]
        denom = den**2 + eta**2
        Oz = -2 * sum_m prod/denom          Lz = sum_m den*prod/denom

    The engine writes `hbar**2 * v` where `v = dH/d(hbar k)`; we use `dH/dk`
    directly, which is the same product with one fewer round trip through hbar.

    Parameters
    ----------
    eta : level repulsion cutoff in eV.  Needed where bands touch -- at
        ``du = 0`` the ABC bands are degenerate at the Dirac points and the
        curvature is a delta function there.

    Returns
    -------
    ``(E, Oz, Lz)`` of shapes ``(Nk, nb)``, ``(Nk, nb)``, ``(Nk, nb)``.
    `E` is in eV, `Oz` in m^2, `Lz` in eV*m^2.
    """
    kx = np.atleast_1d(np.asarray(kx, float)).ravel()
    ky = np.atleast_1d(np.asarray(ky, float)).ravel()
    nk = kx.size

    E_out = Oz_out = Lz_out = None
    for s in range(0, nk, KUBO_CHUNK):
        sl = slice(s, min(s + KUBO_CHUNK, nk))
        H = hamiltonian(model, kx[sl], ky[sl], du, p, valley)
        Vx, Vy = dH_dk(model, kx[sl], ky[sl], du, p, valley)

        E, Psi = np.linalg.eigh(H)                  # (nc, nb), (nc, nb, nb)
        # into the eigenbasis: v(n,m) = <n| V |m>
        vx = np.einsum("kin,kij,kjm->knm", Psi.conj(), Vx, Psi)
        vy = np.einsum("kin,kij,kjm->knm", Psi.conj(), Vy, Psi)

        den = E[:, :, None] - E[:, None, :]         # (nc, n, m)
        prod = np.imag(vx * np.swapaxes(vy, -1, -2))   # Im[vx(n,m) vy(m,n)]
        ratio = prod / (den**2 + eta**2)

        # the n = m terms vanish anyway (vx, vy diagonal entries are real for a
        # Hermitian operator) but den = 0 there, so zero them explicitly
        idx = np.arange(E.shape[-1])
        ratio[:, idx, idx] = 0.0

        if E_out is None:
            E_out = np.empty((nk, E.shape[1]))
            Oz_out = np.empty_like(E_out)
            Lz_out = np.empty_like(E_out)
        E_out[sl] = E
        Oz_out[sl] = -2.0 * ratio.sum(axis=-1)
        Lz_out[sl] = (den * ratio).sum(axis=-1)

    return E_out, Oz_out, Lz_out


# --------------------------------------------------------- band structure

def band_structure(model, du, p: ABCParams, kmax_ev=0.40, nk=400, eta=2e-3,
                   nband_keep=2):
    """Everything the engine needs, in its units, for one model and one dU.

    Parameters
    ----------
    kmax_ev : patch half-width expressed as ``hbar*v*k`` in eV, matching the
        convention in `scripts/export_zerofield.py`.
    nband_keep : how many bands around charge neutrality to keep.  2 is the
        low-energy pair (the A1/B3-derived bands); `None` keeps all of them.

    Returns a dict with the engine's key names and units: `E_K`, `E_Kp` in meV
    shaped ``(nbands, Nk)``, `Oz_*` in m^2, `Lz_*` in meV*m^2, `kpoints` in
    Angstrom^-1, `vol_M` in m^2.
    """
    kmax = kmax_ev / p.hbar_v
    kx, ky, dk = kmesh(kmax, nk)
    vol_M = vol_M_for(nk, nk, dk)

    out = {"model": model, "dU_eV": du, "nk1": nk, "nk2": nk,
           "vol_M": vol_M, "kmax_ev": kmax_ev, "eta_eV": eta,
           "dk_m": dk,
           # the patch is a finite region, not a torus.  The engine reads this
           # key to switch its contour tracer to periodic=False; see
           # semiclassical.py::_periodic in the engine tree.
           "extended_zone": 1}

    nb_all = 2 if model.startswith("two_band") else 6
    if nband_keep is None:
        keep = np.arange(nb_all)
    else:
        lo = nb_all // 2 - nband_keep // 2
        keep = np.arange(lo, lo + nband_keep)

    for valley, tag in ((+1, "K"), (-1, "Kp")):
        E, Oz, Lz = kubo(model, kx, ky, du, p, valley, eta)
        out[f"E_{tag}"] = (E[:, keep] * 1e3).T            # meV, (nbands, Nk)
        out[f"Oz_{tag}"] = Oz[:, keep].T                  # m^2
        out[f"Lz_{tag}"] = (Lz[:, keep] * 1e3).T          # meV*m^2

    out["kpoints"] = np.column_stack([kx, ky]) / 1e10     # Angstrom^-1
    out["bands_kept"] = keep
    return out


def save_bandstructure(path, bs, params=None):
    """Write a band structure in the engine's own `.mat` layout.

    Interop only -- the drivers here call the engine's functions directly.  With
    this file the engine's own `semiclassical.py` can be run on ABC data:

        calctype = 'isoenergy'
        inputdata = '<this file>'

    `extended_zone = 1` travels with the data, which is how the engine learns
    not to tile the surface when tracing contours.
    """
    from scipy.io import savemat

    data = {k: v for k, v in bs.items()}
    data["readme"] = (
        "ABC trilayer graphene zero-field band structure, written in the "
        "layout of hofstadter/source_code/semiclassical.  E in meV, Oz in m^2, "
        "Lz in meV*m^2, kpoints in Angstrom^-1, vol_M in m^2.  The k-mesh is a "
        "square isotropic patch of continuum k-space centred on the valley "
        "point, NOT a Brillouin zone: vol_M is set so that cell_area = dk^2 "
        "and extended_zone = 1 marks the surface as non-periodic so the "
        "contour tracer does not tile it.")
    savemat(str(path), {"results": data, "params": params or {}},
            do_compression=True)
    return path
