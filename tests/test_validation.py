"""Validation of the ABC trilayer Landau-level machinery.

Run with:  python3 -m pytest tests -q        (or just: python3 tests/test_validation.py)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import bands, dos, landau, matio, plotstyle  # noqa: E402
from abc_ll.params import ABCParams, DEFAULT, nphi  # noqa: E402

P = DEFAULT


# ------------------------------------------------------- ladder conventions
def test_monolayer_landau_levels():
    """H = v(xi pi_x sx + pi_y sy) must give E_n = +- omega sqrt(n)."""
    n_ll, B = 200, 7.0
    omega = float(P.omega(B))
    for valley in (+1, -1):
        vpi, vpid = landau.pi_matrices(n_ll, valley, omega)
        z = np.zeros((n_ll, n_ll))
        h = np.block([[z, vpid], [vpi, z]])
        assert np.allclose(h, h.conj().T), "monolayer H not Hermitian"
        e = np.sort(np.linalg.eigvalsh(h))
        # lowest few positive levels, away from the truncation edge
        pos = np.sort(e[e > 1e-12])[:8]
        expect = omega * np.sqrt(np.arange(1, 9))
        assert np.allclose(pos, expect, rtol=1e-10), (pos, expect)
        assert np.sum(np.abs(e) < 1e-10) >= 1, "missing the n=0 zero mode"
    # scale check against the textbook 36.3 meV sqrt(B[T]) for v = 1e6 m/s
    assert abs(float(P.omega(1.0)) * 1e3 - 36.9) < 0.3


def test_hermiticity():
    for model in landau.MODELS:
        for valley in (+1, -1):
            h = landau.build(model, 5.0, 0.12, P, n_ll=60, valley=valley)
            assert np.allclose(h, h.conj().T, atol=1e-14), model


# ------------------------------------------------------------ zero modes
def test_zero_mode_multiplet():
    """Minimal two-band ABC at du=0: exactly three zero levels per valley.

    Gives the famous 12-fold (3 orbital x 2 spin x 2 valley) zero-energy
    multiplet of rhombohedral trilayer graphene.
    """
    for valley in (+1, -1):
        lv = landau.solve_levels("two_band_minimal", 8.0, 0.0, P,
                                 n_ll=200, valley=valley, window=0.3)
        assert np.sum(np.abs(lv.energy) < 1e-9) == 3, lv.energy[:6]


def test_cubic_level_formula():
    """Minimal two-band levels must be E = +- omega^3 sqrt(n(n-1)(n-2)) / gamma1^2."""
    B, n_ll = 10.0, 160
    omega = float(P.omega(B))
    lv = landau.solve_levels("two_band_minimal", B, 0.0, P, n_ll=n_ll,
                             valley=+1, window=0.3)
    n = np.arange(3, 40)
    expect = omega**3 * np.sqrt(n * (n - 1) * (n - 2)) / P.gamma1**2
    expect = np.sort(expect[expect < 0.3])[:6]
    got = np.sort(lv.energy[lv.energy > 1e-9])[:6]
    assert np.allclose(got, expect, rtol=1e-10), (got, expect)


def test_displacement_valley_polarises_zero_multiplet():
    """du splits the zero multiplet *by valley*, not by orbital index.

    In the minimal two-band model the three zero modes of valley K live
    entirely on B3 and so sit at exactly +du/2, while those of K' live on A1
    and sit at -du/2.  The 12-fold zero multiplet therefore splits into two
    fully valley-polarised sextets separated by du -- it does not fan out.
    """
    du = 0.1
    for valley, sign in ((+1, +1.0), (-1, -1.0)):
        lv = landau.solve_levels("two_band_minimal", 8.0, du, P, n_ll=200,
                                 valley=valley, window=0.3)
        at = lv.energy[np.abs(lv.energy - sign * du / 2) < 1e-9]
        assert at.size == 3, f"valley {valley}: {at.size} levels at {sign*du/2}"
        # and nothing at the opposite sign
        other = lv.energy[np.abs(lv.energy + sign * du / 2) < 1e-9]
        assert other.size == 0, f"valley {valley}: unexpected {other}"


# ----------------------------------------------------- model consistency
def test_two_band_matches_six_band_at_zero_field():
    """At small k the two-band projection must track the six-band bands.

    Checks both the minimal pair (gamma0, gamma1 only) and the full pair, and
    asserts that the *extended* two-band model is the better approximation to
    the full six-band model than the minimal one is.
    """
    du = 0.1
    k = np.linspace(0.0, 0.12, 60) / P.hbar_v  # up to hbar v k = 0.12 eV

    for mm, tb in (("six_band_minimal", "two_band_minimal"),):
        e6 = bands.low_energy_pair(mm, k, 0.0 * k, du, P)
        e2 = bands.low_energy_pair(tb, k, 0.0 * k, du, P)
        err = max(np.max(np.abs(e6[0] - e2[0])), np.max(np.abs(e6[1] - e2[1])))
        assert err < 8e-3, f"{tb} vs {mm}: max error {err*1e3:.2f} meV"

    e6 = bands.low_energy_pair("six_band", k, 0.0 * k, du, P)
    e_ext = bands.low_energy_pair("two_band", k, 0.0 * k, du, P)
    e_min = bands.low_energy_pair("two_band_minimal", k, 0.0 * k, du, P)
    err_ext = max(np.max(np.abs(e6[i] - e_ext[i])) for i in (0, 1))
    err_min = max(np.max(np.abs(e6[i] - e_min[i])) for i in (0, 1))
    assert err_ext < err_min, (
        f"extended two-band ({err_ext*1e3:.2f} meV) should beat minimal "
        f"({err_min*1e3:.2f} meV)")


def test_landau_matches_zero_field_dispersion():
    """Cross-check landau.py against bands.py via Onsager quantisation.

    At large Landau index the level E_n must equal the B=0 band energy at
    ``hbar v k = omega sqrt(n - 1)``.  The index offset of -1 is the
    Berry-phase/index shift of the cubic band touching; with it the agreement
    goes as 1/(n-1)^2.  This ties the ladder normalisation in landau.py to the
    independent scalar-k code in bands.py.
    """
    B = 2.0
    omega = float(P.omega(B))
    lv = landau.solve_levels("two_band_minimal", B, 0.0, P, n_ll=400,
                             valley=+1, window=0.3)
    e_ll = np.sort(lv.energy[lv.energy > 1e-9])
    n = np.arange(3, 3 + e_ll.size)
    k = omega * np.sqrt(n - 1.0) / P.hbar_v
    _, e_up = bands.low_energy_pair("two_band_minimal", k, 0.0 * k, 0.0, P)
    big = n > 25
    rel = np.max(np.abs(e_ll[big] - e_up[big]) / e_up[big])
    assert rel < 1e-3, rel


# ------------------------------------------------------------ truncation
def test_basis_convergence():
    """Retained levels must be stable against doubling the oscillator basis."""
    for model in landau.MODELS:
        lv, shift = landau.converged_levels(model, 4.0, 0.1, P, valley=+1,
                                            window=0.15)
        assert shift < 1e-7, f"{model}: residual shift {shift:.2e} eV"
        assert len(lv) > 5, f"{model}: only {len(lv)} levels retained"


def test_banded_solver_matches_dense():
    for model in ("two_band", "six_band"):
        n_ll = 80
        h = landau.build(model, 6.0, 0.08, P, n_ll=n_ll, valley=-1)
        dense = np.sort(np.linalg.eigvalsh(h))
        lv = landau.solve_levels(model, 6.0, 0.08, P, n_ll=n_ll, valley=-1,
                                 window=0.1, tail_tol=np.inf)
        sel = np.sort(dense[np.abs(dense) <= 0.1])
        assert np.allclose(sel, lv.energy, atol=1e-10), model


# ------------------------------------------------------------------- DOS
def test_density_and_dos_consistency():
    """n(mu) must be the integral of DOS(mu), and n(0) must vanish."""
    br = dos.Broadening(gamma0=2e-3, scaling="sqrtB", lineshape="lorentzian")
    B = 6.0
    lv = np.concatenate([
        landau.solve_levels("two_band", B, 0.1, P, n_ll=200, valley=v).energy
        for v in (+1, -1)])
    assert abs(dos.density(0.0, lv, B, br)) < 1e-9

    mu = np.linspace(-0.05, 0.05, 20001)
    d = dos.dos(mu, lv, B, br)
    n_int = np.cumsum(d) * (mu[1] - mu[0])
    n_int -= n_int[len(mu) // 2]
    n_cls = dos.density(mu, lv, B, br)
    err = np.max(np.abs(n_int - n_cls)) / np.max(np.abs(n_cls))
    assert err < 1e-3, err


def test_solve_mu_roundtrip():
    br = dos.Broadening(gamma0=1e-3)
    B = 9.0
    lv = np.concatenate([
        landau.solve_levels("two_band", B, 0.1, P, n_ll=200, valley=v).energy
        for v in (+1, -1)])
    for nu in (2.0, 4.0, 7.5, -3.0):
        n_t = nu * nphi(B)
        mu = dos.solve_mu(n_t, lv, B, br)
        assert np.isfinite(mu)
        assert abs(dos.density(mu, lv, B, br) - n_t) / abs(n_t) < 1e-6


def test_sharp_limit_filling_steps():
    """Charge neutrality half-fills the 12-fold zero multiplet.

    With n measured from the CNP, the zero multiplet spans nu = -6 .. +6, so
    mu stays pinned at E=0 until nu exceeds +6, then jumps to the first
    cubic level.
    """
    br = dos.Broadening(gamma0=1e-6, scaling="const")
    B = 10.0
    lv = np.concatenate([
        landau.solve_levels("two_band_minimal", B, 0.0, P, n_ll=160, valley=v).energy
        for v in (+1, -1)])
    # 6 levels at E=0 (3 orbital x 2 valley); spin is the g_spin=2 factor,
    # so the multiplet holds 12 states per flux quantum.
    assert np.sum(np.abs(lv) < 1e-12) == 6

    mu = dos.solve_mu(5.5 * nphi(B), lv, B, br)
    assert abs(mu) < 1e-5, f"mu={mu} should still be inside the zero multiplet"
    mu = dos.solve_mu(6.5 * nphi(B), lv, B, br)
    omega = float(P.omega(B))
    e1 = omega**3 * np.sqrt(3 * 2 * 1) / P.gamma1**2
    assert abs(mu - e1) < 1e-5, f"mu={mu} should sit on the first cubic level {e1}"


# -------------------------------------------- folding, auto-solve, tracing
def test_folded_equals_analytic_at_zero_displacement():
    """At du = 0 the analytic cubic two-band model is *exactly* the Loewdin
    folding of the six-band model, not an approximation to it.

    The 4x4 dimer block of the ABC chain has determinant gamma1^4 with no
    momentum dependence, so the A1-B3 element of its inverse is exactly
    -(v pi^dg)/gamma1^2 and the third-order Schrieffer-Wolff result
    (v pi^dg)^3 / gamma1^2 is exact.  This pins the sign and prefactor of the
    cubic term to machine precision.
    """
    for B in (3.0, 9.0):
        for valley in (+1, -1):
            a = landau.h_two_band(B, 0.0, P, n_ll=60, valley=valley,
                                  minimal=True)
            b = landau.h_two_band_folded(B, 0.0, P, n_ll=60, valley=valley,
                                         minimal=True)
            assert np.max(np.abs(a - b)) < 1e-14, np.max(np.abs(a - b))


def test_folded_beats_analytic_under_displacement():
    """With du on, the analytic model's second-order treatment of the layer
    potential costs it an order of magnitude against the exact folding."""
    def err(test, ref, k=6):
        r = np.sort(ref[np.argsort(np.abs(ref))[:k]])
        t = np.sort(test[np.argsort(np.abs(test))[:k]])
        return np.max(np.abs(r - t))

    for du in (0.05, 0.10):
        ref = landau.solve_auto("six_band_minimal", 6.0, du, P, valley=+1,
                                window=0.25).energy
        fold = landau.solve_auto("two_band_folded_minimal", 6.0, du, P,
                                 valley=+1, window=0.25).energy
        ana = landau.solve_auto("two_band_minimal", 6.0, du, P, valley=+1,
                                window=0.25).energy
        e_fold, e_ana = err(fold, ref), err(ana, ref)
        assert e_fold < 1e-2, f"du={du}: folded error {e_fold*1e3:.1f} meV"
        assert e_fold < 0.3 * e_ana, (du, e_fold, e_ana)

def test_solve_auto_recovers_levels_a_small_basis_loses():
    """The regression this guards: an undersized basis silently discards
    genuine levels, which shifts every branch index after them."""
    B, du, window = 1.5, 0.20, 0.12
    small = landau.solve_levels("six_band", B, du, P, n_ll=84, valley=-1,
                                window=window)
    auto = landau.solve_auto("six_band", B, du, P, valley=-1, window=window)
    assert small.n_dropped > landau.DROP_FLOOR, small.n_dropped
    assert auto.n_dropped <= landau.DROP_FLOOR, auto.n_dropped
    assert len(auto) > len(small) + 20, (len(auto), len(small))
    # and the recovered set is stable in a doubled basis
    landau.solve_auto("six_band", B, du, P, valley=-1, window=window,
                      verify=True)


def test_trace_branches_is_continuous_across_zero():
    """Levels crossing E = 0 must not break the branch tracing."""
    x = np.linspace(0.0, 1.0, 61)
    cols = [np.sort(np.array([-0.5 + t, 0.3 - 0.8 * t])) for t in x]
    rows = plotstyle.trace_branches(cols, gap=0.2)
    assert rows.shape == (2, 61)
    assert np.all(np.isfinite(rows))
    step = np.max(np.abs(np.diff(rows, axis=1)))
    assert step < 0.03, f"max step {step} -- a branch jumped"


def test_trace_branches_on_real_spectrum():
    """No level should jump by more than a few meV between adjacent du."""
    du = np.linspace(0.0, 0.20, 81)
    cols = [landau.solve_auto("six_band", 6.0, d, P, valley=+1,
                              window=0.1).energy for d in du]
    rows = plotstyle.trace_branches(cols)
    steps = np.abs(np.diff(rows, axis=1))
    worst = np.nanmax(steps)
    assert worst < 4e-3, f"branch jumped by {worst*1e3:.2f} meV"


def test_fast_solver_matches_eigenvector_solver():
    """For the six-band models the eigenvalue-only path must reproduce the
    eigenvector path exactly, including the level count.

    The fast path drops truncation artefacts by matching levels across two
    basis sizes instead of inspecting eigenvectors.  This is the test that
    licenses using it for production maps.
    """
    worst, cases = 0.0, 0
    for model in ("six_band",):
        for B in (1.0, 2.0, 6.0, 14.0):
            for du in (0.0, 0.08, 0.20):
                for xi in (+1, -1):
                    a = landau.solve_auto(model, B, du, P, valley=xi,
                                          window=0.12, fast=True).energy
                    b = landau.solve_auto(model, B, du, P, valley=xi,
                                          window=0.12, fast=False).energy
                    assert len(a) == len(b), (model, B, du, xi, len(a), len(b))
                    if a.size:
                        worst = max(worst, float(np.max(np.abs(a - b))))
                    cases += 1
    assert cases == 24
    assert worst < 1e-12, worst


def test_only_the_production_model_uses_the_fast_path():
    """Guard the FAST_SAFE rule: elsewhere the truncation artefacts are exact
    zero modes, so matching across basis sizes cannot see them."""
    assert landau.FAST_SAFE == frozenset({"six_band"})
    # the blindness fallback must also fire on its own
    blind = landau.solve_auto("six_band_minimal", 6.0, 0.0, P, valley=+1,
                              window=0.12, fast=True)
    ref = landau.solve_auto("six_band_minimal", 6.0, 0.0, P, valley=+1,
                            window=0.12, fast=False)
    assert len(blind) == len(ref)
    auto = landau.solve_auto("two_band", 6.0, 0.0, P, valley=+1, window=0.12)
    slow = landau.solve_auto("two_band", 6.0, 0.0, P, valley=+1, window=0.12,
                             fast=False)
    forced = landau.solve_auto("two_band", 6.0, 0.0, P, valley=+1, window=0.12,
                               fast=True)
    assert len(auto) == len(slow), "default must pick the eigenvector path"
    # even when forced, the blindness fallback hands back to the slow path
    assert len(forced) == len(slow), (len(forced), len(slow))
    assert np.allclose(forced.energy, slow.energy, atol=1e-12)


def test_asymmetric_energy_range():
    """e_range must select exactly the levels a symmetric window would, and
    the density/DOS must not depend on how wide the unused side is."""
    full = landau.solve_auto("six_band", 8.0, 0.1, P, valley=+1,
                             window=0.2).energy
    part = landau.solve_auto("six_band", 8.0, 0.1, P, valley=+1,
                             e_range=(-0.02, 0.14)).energy
    expect = full[(full >= -0.02) & (full <= 0.14)]
    assert len(part) == len(expect), (len(part), len(expect))
    assert np.allclose(part, expect, atol=1e-12)


def test_dos_map_point_is_window_independent():
    """With a Gaussian lineshape, DOS(E_F) must not depend on how much of the
    far valence band is kept -- which is what makes the narrow asymmetric
    energy range used by the map legitimate."""
    gate = dos.moire_gate()
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    for B, du in ((4.0, 0.08), (12.0, 0.18)):
        mu_a, d_a, _, _ = dos.fermi_level_and_dos("six_band", B, du, P, gate, br)
        lv = dos.levels_both_valleys("six_band", B, du, P, e_range=(-0.30, 0.30))
        mu_b = dos.solve_mu(gate.density(du), lv, B, br, bracket=(-0.28, 0.28))
        d_b = float(dos.dos(mu_b, lv, B, br))
        assert abs(mu_a - mu_b) < 2e-3, (B, du, mu_a, mu_b)
        assert abs(d_a - d_b) / d_b < 1e-6, (B, du, d_a, d_b)


def test_moire_gate_calibration():
    g = dos.moire_gate()
    assert abs(g.n_max / 1e4 - 5.172e12) / 5.172e12 < 1e-3
    assert abs(dos.moire_cell_area() * 1e18 - 77.34) < 0.05
    # nu at filling 4 and 15 T
    nu = g.n_max / (2.41799e14 * 15.0)
    assert abs(nu - 14.26) < 0.02, nu


def test_lorentzian_tails_do_not_converge():
    """Documents why the maps default to a Gaussian lineshape.

    A Lorentzian's 1/E^2 tails make the in-gap DOS depend on how many distant
    levels are retained; a Gaussian's does not.  If this ever starts passing
    for the Lorentzian, the range-narrowing optimisation could be used with
    it too.
    """
    gate = dos.moire_gate()
    B, du = 4.0, 0.08
    out = {}
    for ls in ("lorentzian", "gaussian"):
        br = dos.Broadening(gamma0=1e-3, lineshape=ls)
        vals = []
        for hi in (0.14, 0.9):
            lo = -0.02 if hi < 0.2 else -(hi - 0.12)
            lv = dos.levels_both_valleys("six_band", B, du, P,
                                         e_range=(lo, hi))
            mu = dos.solve_mu(gate.density(du), lv, B, br,
                              bracket=(-0.015, 0.13))
            vals.append(float(dos.dos(mu, lv, B, br)))
        out[ls] = vals[0] / vals[1]
    assert abs(out["gaussian"] - 1.0) < 1e-9, out["gaussian"]
    assert abs(out["lorentzian"] - 1.0) > 0.05, out["lorentzian"]


# ---------------------------------------------------------- layer weights
def test_layer_weights_are_normalised_and_complete():
    for model in ("six_band", "two_band"):
        lv = landau.solve_auto(model, 9.0, 0.1, P, valley=+1,
                               e_range=(-0.06, 0.12), layers=True)
        w = lv.layer_weight
        assert w.shape == (len(lv), 3), w.shape
        assert np.all(w >= -1e-12)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
    # the two-band model has no middle layer at all
    lv = landau.solve_auto("two_band", 9.0, 0.1, P, valley=+1,
                           e_range=(-0.06, 0.12), layers=True)
    assert np.allclose(lv.layer_weight[:, 1], 0.0)


def test_displacement_field_layer_polarises_the_zero_modes():
    """The valley splitting of the zero multiplet *is* a layer polarisation.

    At dU = 0 the near-zero level of K sits on the top layer and that of K'
    on the bottom, mirrored; a displacement field then pushes them apart in
    energy without undoing the layer locking.

    The three zero modes of a valley are not equally polarised once
    gamma2/gamma3/gamma4 mix them, so this checks the whole triplet rather
    than any single level.
    """
    for du in (0.0, 0.10):
        w = {}
        for xi in (+1, -1):
            lv = landau.solve_auto("six_band", 10.0, du, P, valley=xi,
                                   e_range=(-0.07, 0.12), layers=True)
            target = (du / 2) * xi
            near = np.argsort(np.abs(lv.energy - target))[:3]
            w[xi] = lv.layer_weight[near]
        # every level of the triplet leans the right way...
        assert np.all(w[+1][:, 2] > w[+1][:, 0]), f"du={du}, K: {w[+1]}"
        assert np.all(w[-1][:, 0] > w[-1][:, 2]), f"du={du}, K': {w[-1]}"
        # ...and the triplet as a whole is strongly polarised
        assert w[+1][:, 2].mean() > 0.6, f"du={du}, K: {w[+1]}"
        assert w[-1][:, 0].mean() > 0.6, f"du={du}, K': {w[-1]}"

        # The two valleys mirror each other exactly at dU = 0.  At finite dU
        # the K triplet sits in the conduction band and the K' triplet in the
        # valence band, so gamma4 and delta -- the electron-hole asymmetry
        # terms -- break the mirror, by ~8 percentage points at dU = 0.1 eV.
        mirror = abs(w[+1][:, 2].mean() - w[-1][:, 0].mean())
        assert mirror < (1e-9 if du == 0.0 else 0.15), (du, mirror)
    assert mirror > 0.02, (
        "expected gamma4/delta to break the valley mirror at finite dU; "
        f"got {mirror}")


def test_layer_resolved_dos_sums_to_total():
    gate = dos.moire_gate()
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    for B, du in ((3.0, 0.02), (9.0, 0.08), (15.0, 0.16)):
        mu, d, dl, _ = dos.fermi_level_and_dos("six_band", B, du, P, gate, br,
                                               layers=True)
        assert dl.shape == (3,)
        assert abs(dl.sum() - d) / d < 1e-12, (B, du, dl.sum(), d)
        assert np.all(dl >= 0)


def test_layers_do_not_change_the_energies():
    """Requesting layer weights must not perturb the spectrum it reports."""
    a = landau.solve_auto("six_band", 7.0, 0.14, P, valley=-1,
                          e_range=(-0.05, 0.13), layers=True)
    b = landau.solve_auto("six_band", 7.0, 0.14, P, valley=-1,
                          e_range=(-0.05, 0.13), layers=False)
    assert len(a) == len(b)
    assert np.allclose(a.energy, b.energy, atol=1e-12)


# ------------------------------------------- negative displacement field
def test_inversion_symmetry():
    """Rhombohedral stacking is centrosymmetric, so dU -> -dU is a symmetry.

    Inversion about the middle layer maps A1 <-> B3 and K <-> K', so the
    spectrum at -dU equals that at +dU with the valleys swapped, and the
    layer weights reverse.  This is a strong check on the whole Hamiltonian
    assembly: a sign slip in gamma3 or gamma4, or a mislabelled layer, breaks
    it immediately.
    """
    worst_e = worst_w = 0.0
    for B in (4.0, 11.0):
        for du in (0.03, 0.10, 0.20):
            for xi in (+1, -1):
                a = landau.solve_auto("six_band", B, du, P, valley=xi,
                                      e_range=(-0.13, 0.13), layers=True)
                b = landau.solve_auto("six_band", B, -du, P, valley=-xi,
                                      e_range=(-0.13, 0.13), layers=True)
                assert len(a) == len(b), (B, du, xi, len(a), len(b))
                worst_e = max(worst_e, float(np.max(np.abs(a.energy - b.energy))))
                worst_w = max(worst_w, float(np.max(
                    np.abs(a.layer_weight - b.layer_weight[:, ::-1]))))
    assert worst_e < 1e-13, worst_e
    assert worst_w < 1e-11, worst_w


def test_hole_side_resolves():
    """dU < 0 is hole doping; the solver must open the valence side.

    n(mu) is monotonic with n(0) = 0, so sign(mu) == sign(n) exactly.  An
    earlier version opened the conduction side regardless and could not find
    the Fermi level at all for dU < 0.
    """
    gate = dos.moire_gate()
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    for B in (3.0, 10.0, 15.0):
        for du in (-0.20, -0.09, -0.02):
            mu, d, dl, _ = dos.fermi_level_and_dos("six_band", B, du, P, gate,
                                                   br, layers=True)
            # hole doped means below the *neutrality* energy, which is not
            # zero: delta and gamma4 put it at +6.3 meV at du = 0
            assert np.isfinite(mu) and mu < dos.cnp_energy(du, P), (B, du, mu)
            assert d > 0 and abs(dl.sum() - d) / d < 1e-12
    # and mu sits exactly at the CNP when the gate induces nothing
    mu, _, _, _ = dos.fermi_level_and_dos("six_band", 8.0, 0.0, P, gate, br,
                                          layers=True)
    assert abs(mu - dos.cnp_energy(0.0, P)) < 1e-9, mu


def test_both_sides_are_top_layer_polarised():
    """With a single gate the carriers sit on the top layer either side.

    For dU > 0 the conduction edge is B3-derived (top) and electrons go
    there; for dU < 0 the valence edge is B3-derived and holes go there.
    """
    gate = dos.moire_gate()
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    for du in (-0.15, -0.05, 0.05, 0.15):
        _, _, dl, _ = dos.fermi_level_and_dos("six_band", 10.0, du, P, gate,
                                              br, layers=True)
        top = dl[2] / dl.sum()
        assert top > 0.5, f"dU={du}: top fraction {top:.3f}"
    # at neutrality the two valleys balance exactly
    _, _, dl, _ = dos.fermi_level_and_dos("six_band", 10.0, 0.0, P, gate, br,
                                          layers=True)
    f = dl / dl.sum()
    assert abs(f[0] - f[2]) < 1e-9, f


# ------------------------------------------------------------ mat export
def test_mat_export_roundtrip(tmpdir=None):
    """The .mat files must survive a loadmat round trip, padding included."""
    import os
    import tempfile

    from scipy.io import loadmat

    gate = dos.moire_gate()
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    B = np.array([5.0, 10.0])
    du = np.array([-0.1, 0.0, 0.1])

    mu = np.zeros((2, 3)); d = np.zeros((2, 3)); dl = np.zeros((2, 3, 3))
    lv = [[None] * 3 for _ in range(2)]
    lw = [[None] * 3 for _ in range(2)]
    for i, b in enumerate(B):
        for j, u in enumerate(du):
            mu[i, j], d[i, j], dl[i, j], _ = dos.fermi_level_and_dos(
                "six_band", b, u, P, gate, br, layers=True)
            es, ws = [], []
            for xi in (+1, -1):
                L = landau.solve_auto("six_band", b, u, P, valley=xi,
                                      window=0.08, layers=True)
                es.append(L.energy); ws.append(L.layer_weight)
            e = np.concatenate(es); w = np.concatenate(ws, axis=0)
            o = np.argsort(e)
            lv[i][j] = e[o]; lw[i][j] = w[o]

    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, "m.mat")
        matio.save_dos_map(f1, B, du, mu, d, dl, P, gate, br, "six_band")
        m = loadmat(f1, squeeze_me=True, struct_as_record=False)
        assert np.allclose(m["mu_eV"], mu)
        assert np.allclose(m["dos_layer_m2_eV"], dl)
        assert m["dos_layer_m2_eV"].shape == (2, 3, 3)
        assert m["params"].gamma1 == P.gamma1
        assert abs(m["gate"].n_max_cm2 - gate.n_max / 1e4) < 1
        assert m["broadening"].lineshape == "gaussian"
        assert np.allclose(m["nu"][:, 1], 0.0)          # dU = 0 -> nu = 0
        assert "readme" in m and "units" in m

        f2 = os.path.join(td, "l.mat")
        matio.save_levels(f2, B, du, lv, lw, P, "six_band")
        m2 = loadmat(f2, squeeze_me=True, struct_as_record=False)
        E, C, W = m2["energy_eV"], m2["n_levels"], m2["layer_weight"]
        for i in range(2):
            for j in range(3):
                k = C[i, j]
                assert np.allclose(E[i, j, :k], lv[i][j])
                assert np.all(np.isnan(E[i, j, k:]))
                assert np.allclose(W[i, j, :k].sum(axis=1), 1.0)


def test_charge_neutrality_is_not_at_zero_energy():
    """E = 0 is not charge neutrality once delta and gamma4 are on.

    gamma0/gamma1/gamma2/gamma3 all connect A to B sublattices, so the ABC
    chain stays bipartite and its spectrum is chirally symmetric about zero.
    delta (dimer on-site) and gamma4 (A-A, B-B) break that and push the
    gamma2-induced Dirac points up to +6.3 meV at du = 0, so mu = 0 is
    inside the valence band -- a hole-doped state, not a neutral one.
    """
    # switching both off restores the symmetry exactly
    clean = P.with_(delta=0.0, gamma4=0.0)
    assert abs(dos.cnp_energy(0.0, clean)) < 5e-5, dos.cnp_energy(0.0, clean)

    e0 = dos.cnp_energy(0.0, P)
    assert 5e-3 < e0 < 8e-3, e0
    # each term alone accounts for part of the shift
    assert dos.cnp_energy(0.0, P.with_(delta=0.0)) < e0
    assert dos.cnp_energy(0.0, P.with_(gamma4=0.0)) < e0

    # inversion symmetry: the neutrality energy is even in du
    for du in (0.05, 0.12, 0.20):
        assert abs(dos.cnp_energy(du, P) - dos.cnp_energy(-du, P)) < 1e-9

    # the zero-mode triplet pins to the Dirac energy, not to zero
    lv = landau.solve_auto("six_band", 2.0, 0.0, P, valley=+1,
                           e_range=(-0.06, 0.06)).energy
    flat = lv[np.abs(lv - e0) < 1e-3]
    assert flat.size >= 3, (e0 * 1e3, lv[:6] * 1e3)


def test_density_is_measured_from_the_cnp():
    """n must vanish at the neutrality energy, not at E = 0."""
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    B = 6.0
    for du in (0.0, 0.01, 0.10):
        e0 = dos.cnp_energy(du, P)
        lv = dos.levels_both_valleys("six_band", B, du, P,
                                     e_range=(e0 - 0.08, e0 + 0.08))
        assert abs(dos.density(e0, lv, B, br, e_cnp=e0)) < 1e-9
        # and at du = 0, mu = 0 is genuinely hole doped
        if du == 0.0:
            n0 = dos.density(0.0, lv, B, br, e_cnp=e0)
            assert n0 < -1e13, n0 / 1e4


def test_cnp_energy_follows_the_model():
    """The neutrality reference must match the model the levels come from.

    The minimal models have delta = gamma4 = 0, so their neutrality really
    is E = 0 at every dU; the full model's is +6.3 meV at dU = 0.  Pairing
    minimal levels with a full-model CNP would offset every filling.
    """
    for du in (0.0, 0.1, 0.2, -0.15):
        assert abs(dos.cnp_energy(du, P, model="six_band_minimal")) < 5e-5
        assert abs(dos.cnp_energy(du, P, model="two_band_minimal")) < 5e-5
    assert dos.cnp_energy(0.0, P, model="six_band") > 5e-3


def test_minimal_model_layer_fraction_is_even_in_filling():
    """Carriers sit on the top layer at both electron and hole doping.

    Two symmetries compose to force this.  Inversion maps the layer-resolved
    DOS of H(-dU) at energy E onto that of H(+dU) at E with the layers
    reversed; chirality-with-layer-reversal maps top(E) onto bottom(-E) at
    fixed dU.  Together they send bottom at filling -f to bottom at +f, so
    every layer fraction is *even* in filling -- not mirrored, which is the
    easy thing to assume.
    """
    gate = dos.moire_gate()
    br = dos.Broadening(gamma0=1e-3, lineshape="gaussian")
    pm = P.minimal()
    for B, du in ((4.0, 0.10), (12.0, 0.18)):
        _, _, dl_p, _ = dos.fermi_level_and_dos("six_band_minimal", B, du,
                                                pm, gate, br, layers=True)
        _, _, dl_m, _ = dos.fermi_level_and_dos("six_band_minimal", B, -du,
                                                pm, gate, br, layers=True)
        fp, fm = dl_p / dl_p.sum(), dl_m / dl_m.sum()
        assert np.allclose(fp, fm, atol=1e-9), (B, du, fp, fm)
        assert fp[2] > 0.6, (B, du, fp)      # and it is the top layer


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fail = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            fail += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns)-fail}/{len(fns)} passed")
    sys.exit(1 if fail else 0)
