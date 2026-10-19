"""Density of states, chemical-potential solving, and the DOS(E_F) map.

Physical set-up (single back gate)
----------------------------------
A single back gate fixes *both* the carrier density and the displacement
field, so they are locked together by one lever arm:

    n(du) = kappa * du ,   kappa = n_max / du_max

At fixed gate voltage, ``n`` does **not** change with ``B``; instead the
chemical potential moves to keep ``n`` fixed as the Landau levels fan out.
That is exactly what :func:`solve_mu` does.

Charge neutrality
-----------------
``E = 0`` is taken as the charge-neutrality point.  This is exact for the
minimal models (the A1/B3 on-site energies are ``-du/2`` and ``+du/2`` about a
traceless potential, and the dimer bands sit at ``+-gamma1``), and accurate to
a few meV once ``gamma4``/``delta`` break electron-hole symmetry.
:func:`cnp_shift` quantifies that residual offset.

Densities are per unit area in m^-2 and count **all** flavours (spin x valley):
each Landau level of one valley carries ``g_spin * e B / h`` states, and both
valleys are summed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .params import ABCParams, nphi

# ------------------------------------------------------------- lineshapes
def _lorentzian(x, gamma):
    return gamma / (np.pi * (x**2 + gamma**2))


def _lorentzian_cdf(x, gamma):
    return 0.5 + np.arctan(x / gamma) / np.pi


def _gaussian(x, gamma):
    # gamma is the standard deviation
    return np.exp(-0.5 * (x / gamma) ** 2) / (gamma * np.sqrt(2 * np.pi))


def _gaussian_cdf(x, gamma):
    from scipy.special import erf

    return 0.5 * (1.0 + erf(x / (gamma * np.sqrt(2.0))))


LINESHAPES = {
    "lorentzian": (_lorentzian, _lorentzian_cdf),
    "gaussian": (_gaussian, _gaussian_cdf),
}


@dataclass(frozen=True)
class Broadening:
    """Landau-level broadening model.

    ``gamma0`` is the width in eV at ``B = 1 T`` -- the half-width at half
    maximum for the Lorentzian, the standard deviation for the Gaussian.
    With ``scaling="sqrtB"`` the width grows as ``gamma0 * sqrt(B)``, the
    standard short-range-disorder result for graphene; ``scaling="const"``
    holds it fixed.

    **Lineshape matters more than it looks.**  A Lorentzian's tails fall only
    as 1/E^2, so the DOS *inside a gap* -- which is exactly what the minima of
    a Landau fan are -- picks up contributions from levels arbitrarily far
    away, and never converges as the retained energy range grows (measured:
    still drifting several percent at +-0.9 eV).  The Gaussian has no such
    tail and gives a DOS that is independent of the range kept, to machine
    precision.  Prefer ``"gaussian"`` for DOS maps; if you want the
    Lorentzian, keep a range of at least +-0.3 eV and expect a few percent
    of tail floor in the gaps.
    """

    gamma0: float = 1.0e-3
    scaling: str = "sqrtB"
    lineshape: str = "gaussian"

    def width(self, B):
        if self.scaling == "sqrtB":
            return self.gamma0 * np.sqrt(np.asarray(B, float))
        if self.scaling == "const":
            return self.gamma0 * np.ones_like(np.asarray(B, float))
        raise ValueError(f"unknown scaling {self.scaling!r}")

    @property
    def _pdf(self):
        return LINESHAPES[self.lineshape][0]

    @property
    def _cdf(self):
        return LINESHAPES[self.lineshape][1]


# ------------------------------------------------------------ gate model
@dataclass(frozen=True)
class GateModel:
    """Lever arm locking carrier density to interlayer potential difference.

    Constructed most conveniently from "``n_per_cell`` states per unit cell at
    ``du_max``" via :meth:`from_states_per_cell`.
    """

    kappa: float            # m^-2 per eV of du
    du_max: float = 0.2     # eV, for reporting only
    label: str = ""

    def density(self, du):
        """Carrier density [m^-2] at interlayer potential difference ``du`` [eV]."""
        return self.kappa * np.asarray(du, float)

    @classmethod
    def from_states_per_cell(cls, n_per_cell, cell_area, du_max=0.2, label=""):
        """``n_per_cell`` states per unit cell of area ``cell_area`` [m^2] at ``du_max``."""
        n_max = n_per_cell / cell_area
        return cls(kappa=n_max / du_max, du_max=du_max,
                   label=label or f"{n_per_cell} states/cell at du={du_max} eV")

    @classmethod
    def from_density_at(cls, n_max_m2, du_max=0.2, label=""):
        return cls(kappa=n_max_m2 / du_max, du_max=du_max, label=label)

    @property
    def n_max(self):
        return self.kappa * self.du_max


# ------------------------------------------------------- DOS and filling
@lru_cache(maxsize=4096)
def _cnp_cached(du_key, pkey, kmax, nk, nth, model):
    from . import bands
    from .params import ABCParams

    p = ABCParams(*pkey)
    du = du_key * 1e-9
    k = np.linspace(1e-7, kmax, nk) / p.hbar_v
    th = np.linspace(0.0, 2 * np.pi / 3, nth)        # C3 symmetric
    kx = np.outer(k, np.cos(th))
    ky = np.outer(k, np.sin(th))
    ev, ec = -np.inf, np.inf
    for xi in (+1, -1):
        lo, hi = bands.low_energy_pair(model, kx, ky, du, p, valley=xi)
        ev = max(ev, float(lo.max()))
        ec = min(ec, float(hi.min()))
    return 0.5 * (ev + ec)


def cnp_energy(du, p, kmax=0.40, nk=1200, nth=121, model="six_band"):
    """Charge-neutrality energy at interlayer potential ``du`` [eV].

    **E = 0 is not charge neutrality.**  The ABC chain is bipartite under
    gamma0/gamma1/gamma2/gamma3 (all of which connect A to B sublattices), so
    those alone give a chirally symmetric spectrum centred on zero.  But
    ``delta`` (dimer on-site energy) and ``gamma4`` (which connect A-A and
    B-B) break that, and push the whole low-energy structure off zero: with
    the default parameters the gamma2-induced Dirac points sit at **+6.34
    meV** at ``du = 0``, so mu = 0 is inside the valence band and describes a
    hole-doped state, not a neutral one.

    Neutrality means the lower of the two low-energy bands is full and the
    upper empty, so it is the midpoint of ``(valence max, conduction min)``.
    When a gap is open any energy inside it gives the same density, and the
    midpoint is the natural representative; when the bands touch, it is the
    Dirac energy.

    The offset is bounded -- at most ~1e11 cm^-2, 2% of full scale here --
    and vanishes once the gap is wide enough to contain E = 0, which happens
    by |du| ~ 20 meV.  But that strip is where the zero-mode multiplet lives.

    ``model`` must match the model the levels come from.  The minimal
    models have delta = gamma4 = 0, so their neutrality really is E = 0;
    pairing minimal levels with a full-model CNP would offset every filling
    by ~6 meV worth of density.
    """
    if model.startswith("two_band_folded"):
        model = "six_band_minimal" if model.endswith("minimal") else "six_band"
    pkey = (p.gamma0, p.gamma1, p.gamma2, p.gamma3, p.gamma4, p.delta,
            p.a, p.d0)
    return _cnp_cached(int(round(float(du) * 1e9)), pkey, kmax, nk, nth, model)


def dos(mu, levels, B, broadening: Broadening, g_spin=2):
    """Density of states at energy ``mu`` [states eV^-1 m^-2].

    ``levels`` is the concatenation of the Landau-level energies of *all*
    valleys included; spin is accounted for by ``g_spin``.
    """
    mu = np.asarray(mu, float)
    gam = float(broadening.width(B))
    deg = g_spin * nphi(B)
    return deg * np.sum(broadening._pdf(mu[..., None] - levels, gam), axis=-1)


def density(mu, levels, B, broadening: Broadening, g_spin=2, e_cnp=0.0):
    """Carrier density relative to charge neutrality [m^-2].

    ``n(mu) = integral_{e_cnp}^{mu} DOS(E) dE``, in closed form from the
    lineshape CDF, so ``n(e_cnp) = 0`` identically.

    ``e_cnp`` must be the real neutrality energy -- see :func:`cnp_energy`.
    It is *not* zero in general: delta and gamma4 push the low-energy
    structure off E = 0 by ~6 meV at du = 0.
    """
    mu = np.asarray(mu, float)
    gam = float(broadening.width(B))
    deg = g_spin * nphi(B)
    cdf = broadening._cdf
    return deg * np.sum(cdf(mu[..., None] - levels, gam)
                        - cdf(e_cnp - levels, gam), axis=-1)


def solve_mu(n_target, levels, B, broadening: Broadening, g_spin=2,
             window=0.3, tol=1e-9, max_iter=200, bracket=None, e_cnp=0.0):
    """Chemical potential giving carrier density ``n_target`` [m^-2].

    ``n(mu)`` is monotonically increasing, so a plain bisection is robust.
    Returns ``nan`` if the target density cannot be reached inside the
    bracket (i.e. the level list is too small -- widen it or raise the basis
    size).
    """
    lo, hi = (-window, window) if bracket is None else bracket
    n_lo = density(lo, levels, B, broadening, g_spin, e_cnp)
    n_hi = density(hi, levels, B, broadening, g_spin, e_cnp)
    if not (n_lo <= n_target <= n_hi):
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if density(mid, levels, B, broadening, g_spin, e_cnp) < n_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


#: ABC trilayer aligned to hBN: moire wavelength 9.45 nm.  "Filling 4" means
#: four states (spin x valley) per moire cell, reached at dU = 0.2 eV.
MOIRE_LAMBDA = 9.45e-9


def moire_cell_area(lam=MOIRE_LAMBDA):
    return 0.5 * np.sqrt(3.0) * lam**2


def moire_gate(n_per_cell=4.0, du_max=0.2, lam=MOIRE_LAMBDA) -> GateModel:
    """The single-back-gate lever arm for an ABC/hBN moire sample.

    At lam = 9.45 nm, filling 4 is 5.17e12 cm^-2, which is also what
    unscreened single-gate electrostatics gives for a 0.2 eV drop across the
    trilayer with eps_graphite ~ 3.1 -- so the calibration is self-consistent.
    """
    return GateModel.from_states_per_cell(
        n_per_cell, moire_cell_area(lam), du_max=du_max,
        label=f"{n_per_cell:g} states per {lam*1e9:.2f} nm moire cell "
              f"at dU = {du_max} eV")


def levels_both_valleys(model, B, du, p, window=None, valleys=(+1, -1),
                        e_range=None):
    """Landau levels of all valleys, concatenated and sorted."""
    from . import landau

    return np.sort(np.concatenate([
        landau.solve_auto(model, B, du, p, valley=xi, window=window or 0.3,
                          e_range=e_range).energy
        for xi in valleys]))


def levels_and_layers(model, B, du, p, window=None, valleys=(+1, -1),
                      e_range=None):
    """Landau levels of all valleys plus their per-layer weights.

    Returns ``(energy, layer_weight)`` with shapes ``(n,)`` and ``(n, 3)``,
    sorted by energy.  Requires eigenvectors, so this is the expensive path.
    """
    from . import landau

    es, ws = [], []
    for xi in valleys:
        lv = landau.solve_auto(model, B, du, p, valley=xi,
                               window=window or 0.3, e_range=e_range,
                               layers=True)
        es.append(lv.energy)
        ws.append(lv.layer_weight)
    e = np.concatenate(es)
    w = np.concatenate(ws, axis=0)
    o = np.argsort(e)
    return e[o], w[o]


def dos_layers(mu, levels, layer_weight, B, broadening: Broadening, g_spin=2):
    """DOS at ``mu`` resolved by layer: a (3,) array summing to :func:`dos`.

    Layer 0 is the bottom (on-site potential -dU/2), layer 2 the top (+dU/2).
    """
    gam = float(broadening.width(B))
    deg = g_spin * nphi(B)
    shape = broadening._pdf(float(mu) - np.asarray(levels), gam)
    return deg * (np.asarray(layer_weight) * shape[:, None]).sum(axis=0)


def fermi_level_and_dos(model, B, du, p, gate: GateModel,
                        broadening: Broadening, g_spin=2, e_hi0=0.14,
                        max_grow=5, margin=0.02, valleys=(+1, -1),
                        layers=False, e_max=0.45):
    """Chemical potential and DOS(E_F) at one (B, dU), at fixed gate voltage.

    The density is set by the gate and does **not** move with B; the chemical
    potential does, which is what produces the Landau fan.

    Only levels within a few linewidths of the interval ``[0, mu]`` matter:
    in ``n(mu) = sum [CDF(mu-E) - CDF(-E)]`` a level far above ``mu``
    contributes nothing and one far below contributes equally at both ends.
    So the solve uses an **asymmetric** energy range that brackets ``[0, mu]``
    rather than a symmetric window -- which is what makes the map affordable,
    since it both shrinks the basis and cuts the number of levels solved for.
    The range is grown until ``mu`` sits comfortably inside it.

    (With a Lorentzian lineshape the excluded levels still leak in through
    their tails, at the ~1% level here; the Gaussian lineshape has no tail.)

    With ``layers=True`` the per-layer DOS is also returned, which needs
    eigenvectors and costs roughly 5-20x more.

    The density is measured from the true charge-neutrality energy
    :func:`cnp_energy`, which is **not** E = 0 -- delta and gamma4 shift it
    to +6.3 meV at du = 0 with the default parameters.

    Returns ``(mu, dos, dos_by_layer_or_None, e_range_used)``.  Note ``mu``
    is still reported on the absolute energy axis, so at neutrality it comes
    back as ``cnp_energy(du, p)``, not zero.
    """
    n_target = gate.density(du)
    e0 = cnp_energy(du, p, model=model)   # neutrality is NOT at E = 0
    pad = max(0.02, 12.0 * float(broadening.width(B)))
    reach = max(e_hi0, pad)
    # n(mu) is monotonic and vanishes at e0, so sign(mu - e0) ==
    # sign(n_target) exactly.  Open the range on that side only -- on the
    # hole side (dU < 0 for a single gate) that is the *valence* side.
    electrons = n_target >= 0
    lo, hi = (e0 - pad, e0 + reach) if electrons else (e0 - reach, e0 + pad)
    for _ in range(max_grow):
        if layers:
            lv, lw = levels_and_layers(model, B, du, p, valleys=valleys,
                                       e_range=(lo, hi))
        else:
            lv, lw = levels_both_valleys(model, B, du, p, valleys=valleys,
                                         e_range=(lo, hi)), None
        mu = solve_mu(n_target, lv, B, broadening, g_spin,
                      bracket=(lo + margin, hi - margin), e_cnp=e0)
        if np.isfinite(mu) and lo + margin < mu < hi - margin:
            dl = (dos_layers(mu, lv, lw, B, broadening, g_spin)
                  if layers else None)
            return mu, float(dos(mu, lv, B, broadening, g_spin)), dl, (lo, hi)
        # Bounded growth.  Unbounded, this reaches ~1.8 eV in five steps,
        # and the basis needed to represent that at low field is large
        # enough to get the worker OOM-killed -- which surfaces as an
        # inscrutable BrokenProcessPool rather than an error anyone can act
        # on.  Fail loudly at e_max instead.
        if electrons:
            hi = min(hi * 1.6 + pad, e_max)
        else:
            lo = max(lo * 1.6 - pad, -e_max)
        if hi >= e_max and lo <= -e_max:
            break
    raise RuntimeError(
        f"{model} at B={B} T, dU={du} eV: Fermi level did not settle inside "
        f"[{lo:.3f}, {hi:.3f}] eV (n_target={n_target/1e4:.3e} cm^-2, "
        f"nu={n_target/nphi(B):.1f}). At low field and large |dU| the "
        f"filling is huge and E_F runs off the top of the band window; "
        f"raise e_max, or raise the minimum field.")


def cnp_shift(levels, B, broadening: Broadening, g_spin=2, window=0.3):
    """Diagnostic: where the particle-hole-symmetric point of the level list sits.

    Returns the ``mu`` at which the integrated DOS from ``-window`` equals half
    the total weight in the window.  Far from zero means ``gamma4``/``delta``
    asymmetry is shifting neutrality appreciably.
    """
    gam = float(broadening.width(B))
    cdf = broadening._cdf
    total = np.sum(cdf(window - levels, gam) - cdf(-window - levels, gam))

    lo, hi = -window, window
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        below = np.sum(cdf(mid - levels, gam) - cdf(-window - levels, gam))
        if below < 0.5 * total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
