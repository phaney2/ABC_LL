"""Validation of the semiclassical Onsager path for ABC trilayer graphene.

The useful thing about applying the semiclassical engine to this system is that
`abc_ll.landau` already solves the same Hamiltonian exactly in a magnetic field,
so the semiclassics can be checked rather than assumed.  The sharpest check is
analytic.  For the minimal two-band model at dU = 0 the dispersion is
``E = (hbar v k)^3 / gamma1^2``, so

    S(E) = pi k^2 ,    k = (E gamma1^2)^(1/3) / (hbar v)

and the Onsager condition ``S/(2 pi)^2 = (B/phi0) (n + 1/2 - phi_B/2 pi)`` with
the chirality-3 Berry phase ``phi_B = 3 pi`` gives

    E_n = (hbar v)^3 / gamma1^2 * (2/l_B^2)^(3/2) * (n - 1)^(3/2)

against the exact ``sqrt(n(n-1)(n-2))``.  The two converge as the index grows --
15.5% at n = 3, 0.62% at n = 10, 0.35% at n = 13 -- which is what pins
``Bmultiplier = 1``: the engine's default of 4 would scale every energy by
4^(3/2) = 8, an error no amount of large-n convergence could absorb.

Note the division of labour in that statement.  The *area* is checked
numerically here; the *Berry phase* at dU = 0 is not, because the curvature of a
gapless chiral touching is a delta function at k = 0 that no grid sum can see --
the engine's `enclosedBC` is identically zero there.  Only gapped displacement
fields carry a usable Berry term, and `test_total_flux_is_three_pi` is what
separates the two cases.
"""

import numpy as np
import pytest

from abc_ll import bands as bandmod, landau, semiclassical as sc
from abc_ll.params import DEFAULT as P, LB_1T

PM = P.minimal()

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _engine():
    try:
        return sc.engine()
    except RuntimeError as exc:
        pytest.skip(str(exc))


# --------------------------------------------------------------- derivatives

def test_stencil_matches_analytic_six_band():
    """The six-band H is linear in v*pi, so dH/dk is exact and checkable."""
    kx = np.array([0.3, -0.7, 1.1]) / P.hbar_v
    ky = np.array([0.1, 0.5, -0.9]) / P.hbar_v
    for valley in (+1, -1):
        dx, dy = sc.dH_dk("six_band", kx, ky, 0.1, P, valley)
        # H depends on k only through vpi = hbar_v*(xi*kx + i*ky), and linearly,
        # so the derivative is H(k) - H(0) evaluated at a unit step.
        h0 = bandmod.h_six_band_k(0.0, 0.0, 0.1, P, valley)
        unit = 1.0 / P.hbar_v
        ax = (bandmod.h_six_band_k(unit, 0.0, 0.1, P, valley) - h0) / unit
        ay = (bandmod.h_six_band_k(0.0, unit, 0.1, P, valley) - h0) / unit
        assert np.abs(dx - ax).max() < 1e-8 * np.abs(ax).max()
        assert np.abs(dy - ay).max() < 1e-8 * np.abs(ay).max()


@pytest.mark.parametrize("model", ["two_band_minimal", "two_band", "six_band"])
def test_stencil_is_step_independent(model):
    """Exact for these polynomial orders, so halving h must change nothing."""
    kx = np.array([0.3, -0.7]) / P.hbar_v
    ky = np.array([0.1, 0.5]) / P.hbar_v
    a = sc.dH_dk(model, kx, ky, 0.1, P, +1)
    b = sc.dH_dk(model, kx, ky, 0.1, P, +1, h=0.5e-4 / P.hbar_v)
    for u, v in zip(a, b):
        assert np.abs(u - v).max() < 1e-9 * np.abs(u).max()


# ---------------------------------------------------------------- orbit area

def test_orbit_area_matches_analytic_circle():
    """vol_M round trip: traced area vs pi k^2 for the cubic band."""
    get_energy_resolved_data, _, _ = _engine()
    nk = 301
    bs = sc.band_structure("two_band_minimal", 0.0, PM, kmax_ev=0.40, nk=nk)
    E, Oz, Lz = bs["E_K"][1], bs["Oz_K"][1], bs["Lz_K"][1]

    E_levels = np.linspace(5.0, 150.0, 60)
    area, _, _ = get_energy_resolved_data(
        3.0, E, Oz, Lz, E_levels, bs["vol_M"], nk, nk, False)

    k_an = (E_levels * 1e-3 * PM.gamma1**2) ** (1 / 3) / PM.hbar_v
    ratio = area[:, 0] / (np.pi * k_an**2)
    assert np.abs(ratio - 1.0).max() < 2e-3, f"worst {np.abs(ratio-1).max():.2e}"


# --------------------------------------------------------------- Berry flux

@pytest.mark.parametrize("model,p,du", [
    ("six_band", P, 0.10),
    ("two_band_minimal", PM, 0.10),
])
def test_total_flux_is_three_pi(model, p, du):
    """Gapped chirality-3: the band carries +-3 pi, opposite in the valleys."""
    nk = 301
    bs = sc.band_structure(model, du, p, kmax_ev=0.40, nk=nk)
    kweight = (2 * np.pi) ** 2 / (bs["E_K"].shape[1] * bs["vol_M"])
    fK = bs["Oz_K"].sum(axis=1) * kweight / np.pi
    fKp = bs["Oz_Kp"].sum(axis=1) * kweight / np.pi

    assert np.abs(np.abs(fK) - 3.0).max() < 0.3, f"K flux/pi = {fK}"
    assert np.allclose(fK, -fKp, atol=1e-6), "valleys must carry opposite flux"
    # valence and conduction carry opposite signs
    assert fK[0] * fK[1] < 0


def test_gapless_flux_vanishes():
    """At dU = 0 the curvature is a delta function the grid cannot see.

    Not a numerical failure -- it is why the Berry term is only trustworthy at
    finite displacement field, and why the exporter flags these points.
    """
    bs = sc.band_structure("two_band_minimal", 0.0, PM, kmax_ev=0.40, nk=301)
    kweight = (2 * np.pi) ** 2 / (bs["E_K"].shape[1] * bs["vol_M"])
    assert np.abs(bs["Oz_K"].sum(axis=1) * kweight).max() < 1e-6


def test_enclosed_flux_grows_with_orbit():
    """The Berry correction is energy dependent: bigger orbit, more flux."""
    get_energy_resolved_data, _, _ = _engine()
    nk = 301
    bs = sc.band_structure("six_band", 0.10, P, kmax_ev=0.40, nk=nk)
    E, Oz, Lz = bs["E_K"][1], bs["Oz_K"][1], bs["Lz_K"][1]
    E_levels = np.linspace(60.0, 158.0, 40)
    _, bc, _ = get_energy_resolved_data(
        3.0, E, Oz, Lz, E_levels, bs["vol_M"], nk, nk, False)
    f = np.abs(bc[:, 0]) / np.pi
    assert f[0] < f[-1], "enclosed flux should grow with energy"
    assert f[-1] < 3.2, "cannot exceed the topological total"


# --------------------------------------------------- the Bmultiplier = 1 test

def test_onsager_reproduces_exact_cubic_levels():
    """The analytic check that fixes Bmultiplier = 1.

    At dU = 0 the grid carries no Berry curvature, so `_SB` equals `_S` and the
    fan lands on (n + 1/2)^(3/2).  Supplying the topological 3 pi by hand turns
    that into (n - 1)^(3/2), which is what the exact sqrt(n(n-1)(n-2)) tends to.
    """
    get_energy_resolved_data, onsager_fan_band, _ = _engine()
    nk = 376
    B = 10.0
    # reaching n = 13 needs levels up to ~440 meV, i.e. orbits out to
    # hbar*v*k = 0.41 eV, so the patch has to be wider than the production one
    bs = sc.band_structure("two_band_minimal", 0.0, PM, kmax_ev=0.50, nk=nk)
    E, Oz, Lz = bs["E_K"][1], bs["Oz_K"][1], bs["Lz_K"][1]

    E_levels = np.linspace(2.0, 460.0, 800)
    area, bc, dl = get_energy_resolved_data(
        3.0, E, Oz, Lz, E_levels, bs["vol_M"], nk, nk, False)
    assert np.abs(bc).max() < 1e-9      # gapless: no curvature on the grid

    # hand the engine the topological Berry phase the grid cannot supply
    bc_topological = np.full_like(bc, 3.0 * np.pi)
    ll = onsager_fan_band(np.array([B]), 20, E_levels, area, bc_topological, dl,
                          term_factors=(1.0, 1.0, 0.0), Bmultiplier=1.0)
    got = ll["SB"][0]

    lB = LB_1T / np.sqrt(B)
    pref = PM.hbar_v**3 / PM.gamma1**2 * (2.0 / lB**2) ** 1.5 * 1e3   # meV
    n = np.arange(3, 14)
    exact = pref * np.sqrt(n * (n - 1) * (n - 2))

    # the fan puts level n at (n-1)^(3/2); compare like for like
    sm = got[n]
    assert np.all(np.isfinite(sm)), f"missing levels: {got[:16]}"
    assert np.abs(sm / (pref * (n - 1) ** 1.5) - 1).max() < 1e-3, (
        "fan is not landing on (n-1)^(3/2), so Bmultiplier or the Berry phase "
        f"is off: {sm / (pref * (n - 1)**1.5)}")

    rel = np.abs(sm - exact) / exact
    assert rel[0] < 0.20, f"n=3 off by {rel[0]:.2%}"
    assert rel[7] < 0.01, f"n=10 off by {rel[7]:.2%}"
    assert rel[-1] < 0.005, f"n=13 off by {rel[-1]:.3%}"
    assert np.all(np.diff(rel) < 0), "agreement must improve with n"


def test_bmultiplier_four_is_wrong_here():
    """Guard the one parameter most likely to be copied over from the engine."""
    get_energy_resolved_data, onsager_fan_band, _ = _engine()
    nk = 201
    B = 10.0
    bs = sc.band_structure("two_band_minimal", 0.0, PM, kmax_ev=0.40, nk=nk)
    E, Oz, Lz = bs["E_K"][1], bs["Oz_K"][1], bs["Lz_K"][1]
    E_levels = np.linspace(2.0, 300.0, 400)
    area, bc, dl = get_energy_resolved_data(
        3.0, E, Oz, Lz, E_levels, bs["vol_M"], nk, nk, False)

    lB = LB_1T / np.sqrt(B)
    pref = PM.hbar_v**3 / PM.gamma1**2 * (2.0 / lB**2) ** 1.5 * 1e3
    lo, hi = 20.0, 280.0
    n = np.arange(3, 40)
    exact = pref * np.sqrt(n * (n - 1) * (n - 2))
    n_exact = int(((exact > lo) & (exact < hi)).sum())

    # Count levels in a window rather than matching indices.  lambda enters as
    # S ∝ lambda and E ∝ S^(3/2), so lambda = 4 stretches the whole ladder by
    # 8x; its levels do not merely shift, they thin out.  Index matching cannot
    # express that -- at lambda = 4 the n = 3 level is already at 470 meV and
    # comparing against a NaN would prove nothing.  Level count is the metric
    # the engine's own documentation recommends (nLL/nEx).
    counts = {}
    for lam in (1.0, 4.0):
        ll = onsager_fan_band(np.array([B]), 30, E_levels, area,
                              np.full_like(bc, 3.0 * np.pi), dl,
                              term_factors=(1.0, 1.0, 0.0), Bmultiplier=lam)
        lv = ll["SB"][0]
        lv = lv[np.isfinite(lv)]
        counts[lam] = int(((lv > lo) & (lv < hi)).sum())

    assert counts[1.0] == n_exact, (
        f"lambda=1 gives {counts[1.0]} levels in [{lo}, {hi}] meV, "
        f"exact has {n_exact}")
    assert counts[4.0] < 0.5 * n_exact, (
        f"lambda=4 should thin the fan out; got {counts[4.0]} vs {n_exact}")


# -------------------------------------------------------- valley consistency

def test_valleys_give_the_same_energies():
    """Opposite Berry phase shifts the index, not the spectrum.

    The two valleys carry +-3 pi, so the Onsager condition gives (n - 1) in one
    and (n + 2) in the other: the same ladder read off at indices three apart.
    The exact spectrum is valley degenerate at dU = 0, so anything else here
    would mean a sign error in Oz.
    """
    get_energy_resolved_data, onsager_fan_band, _ = _engine()
    nk = 201
    B = 10.0
    bs = sc.band_structure("two_band_minimal", 0.0, PM, kmax_ev=0.50, nk=nk)
    E_levels = np.linspace(2.0, 400.0, 600)

    levels = {}
    for valley, sign in (("K", +1.0), ("Kp", -1.0)):
        E, Oz, Lz = bs[f"E_{valley}"][1], bs[f"Oz_{valley}"][1], bs[f"Lz_{valley}"][1]
        area, bc, dl = get_energy_resolved_data(
            3.0, E, Oz, Lz, E_levels, bs["vol_M"], nk, nk, False)
        ll = onsager_fan_band(np.array([B]), 30, E_levels, area,
                              np.full_like(bc, sign * 3.0 * np.pi), dl,
                              term_factors=(1.0, 1.0, 0.0), Bmultiplier=1.0)
        levels[valley] = ll["SB"][0]

    # K at index n matches Kp at index n - 3
    a = levels["K"][4:12]
    b = levels["Kp"][1:9]
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.abs(a - b).max() < 1e-6 * np.abs(a).max()


# -------------------------------------------------------------- patch extent

def test_orbits_stay_inside_the_patch():
    """Border-touching orbits are dropped silently, so check for gaps."""
    get_energy_resolved_data, _, _ = _engine()
    nk = 301
    # 0.50 eV is the production patch; the 160 meV orbit at dU = 200 meV
    # reaches 0.4002, which is exactly what a 0.40 patch loses to the border.
    for model, p, du in (("six_band", P, 0.20), ("six_band", P, 0.02)):
        bs = sc.band_structure(model, du, p, kmax_ev=0.50, nk=nk)
        E, Oz, Lz = bs["E_K"][1], bs["Oz_K"][1], bs["Lz_K"][1]
        lo = max(E.min(), -160.0)
        E_levels = np.linspace(lo + 1e-6, 160.0, 120)
        area, _, _ = get_energy_resolved_data(
            3.0, E, Oz, Lz, E_levels, bs["vol_M"], nk, nk, False)
        have = area[:, 0] > 0
        # no hole: once orbits start they must continue to the top of the window
        first = np.argmax(have)
        assert have[first:].all(), (
            f"{model} dU={du}: orbit lost inside the window -- the patch at "
            f"kmax_ev={bs['kmax_ev']} is too small")
