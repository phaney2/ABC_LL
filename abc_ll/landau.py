"""Landau-level Hamiltonians for ABC trilayer graphene, at several levels of
approximation, plus a solver.

Ladder-operator conventions
---------------------------
With ``B`` along +z and electron charge -e, the kinetic momentum is
``pi_vec = p_vec + e A_vec`` and ``[pi_x, pi_y] = -i hbar e B``.  Defining

    a    = (l_B / (sqrt(2) hbar)) (pi_x - i pi_y)
    a^dg = (l_B / (sqrt(2) hbar)) (pi_x + i pi_y)

gives ``[a, a^dg] = 1``.  With the valley-resolved combination
``pi = xi pi_x + i pi_y`` this means

    xi = +1 (K) :   v*pi = +omega a^dg ,   v*pi^dg = +omega a
    xi = -1 (K'):   v*pi = -omega a    ,   v*pi^dg = -omega a^dg

with ``omega = sqrt(2) hbar v / l_B``.  Sanity check: monolayer graphene
``H = v(xi pi_x sx + pi_y sy)`` then has levels ``+-omega sqrt(n)``, which is
the textbook ``+-v sqrt(2 hbar e B n)``.

Everything is assembled as an explicit matrix in a truncated oscillator basis
``|0>, ..., |n_ll-1>``.  That is brute force, but it costs nothing (the
matrices are banded) and it removes all the per-sublattice Landau-index
bookkeeping that makes these calculations error-prone once gamma2/gamma3/gamma4
-- which shift the index by different amounts -- are switched on.

Truncation
----------
``a a^dg`` is wrong in its last entry in a truncated basis, so eigenstates with
weight near ``n = n_ll`` are spurious.  :func:`solve_levels` discards any state
with more than ``tail_tol`` of its weight in the top ``tail_frac`` of Landau
indices, and :func:`converged_levels` repeats the calculation with a larger
basis to confirm stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig_banded

from .params import ABCParams, layer_potentials

MODELS = (
    "two_band_minimal",         # cubic touching + rigid du/2; the textbook model
    "two_band",                 # + gap renormalisation, gamma2, gamma3
    "two_band_folded",          # exact Loewdin folding of the six-band model
    "two_band_folded_minimal",  # ditto, gamma0/gamma1 only
    "six_band_minimal",         # gamma0, gamma1 only
    "six_band",                 # gamma0..gamma4, delta
)

#: number of sublattice components for each model
NBANDS = {
    "two_band_minimal": 2,
    "two_band": 2,
    "two_band_folded": 2,
    "two_band_folded_minimal": 2,
    "six_band_minimal": 6,
    "six_band": 6,
}

#: models whose Hamiltonian is dense (the folding inverts the dimer block, so
#: the result couples all Landau indices) and must use a dense eigensolver.
DENSE_MODELS = frozenset({"two_band_folded", "two_band_folded_minimal"})


# ------------------------------------------------------------------ ladder
def ladder(n_ll: int) -> np.ndarray:
    """Annihilation operator ``a`` on the truncated basis |0..n_ll-1>."""
    return np.diag(np.sqrt(np.arange(1, n_ll)), 1)


def pi_matrices(n_ll: int, valley: int, omega: float):
    """Return ``(v*pi, v*pi^dagger)`` as real matrices in eV."""
    a = ladder(n_ll)
    vpi = omega * a.T if valley > 0 else -omega * a
    return vpi, vpi.T


# ------------------------------------------------------- model Hamiltonians
def h_two_band(B, du, p: ABCParams, n_ll=256, valley=1, minimal=False):
    """Effective two-band (A1, B3) Hamiltonian in the Landau-level basis.

    ``minimal=True`` gives the textbook model: a pure cubic band touching
    plus a rigid ``+-du/2`` layer splitting,

        H = [[ -du/2 ,  (v pi^dg)^3 / gamma1^2 ],
             [  h.c. ,  +du/2                  ]]

    ``minimal=False`` additionally keeps the three corrections that are the
    same order as, or larger than, the features of interest below ~20 meV:

    1. The momentum-dependent renormalisation of the gap that the displacement
       field acquires through the dimer sites (second order in v):
       ``u1 -> u1 - (u1-u2) v^2 pi^dg pi / (2 gamma1^2)`` and likewise for B3.
       This is what turns the band edges into a Mexican hat; dropping it gives
       a monotonic ``sqrt(Delta^2 + c^2 k^6)`` dispersion with no hat at all.
    2. The direct A1-B3 coupling ``gamma2/2``, which splits the cubic band
       touching into separate Dirac points at the ~10 meV scale.
    3. The gamma3-induced term ``-(v v3/gamma1)(pi pi^dg + pi^dg pi)``, which
       arises at second order through either dimer pair and is isotropic in
       |k| (in ABC, unlike bilayer, gamma3 does *not* produce a direct trigonal
       term in the two-band model).
    """
    p_ = p.minimal() if minimal else p
    omega = float(p.omega(B))
    vpi, vpid = pi_matrices(n_ll, valley, omega)
    eye = np.eye(n_ll)
    u1, u2, u3 = layer_potentials(du)
    g1 = p.gamma1

    # off-diagonal A1-B3
    x = vpid @ vpid @ vpid / g1**2
    if not minimal:
        x = x + 0.5 * p_.gamma2 * eye
        x = x - (p_.gamma3 / p_.gamma0) / g1 * (vpi @ vpid + vpid @ vpi)

    h11 = u1 * eye
    h22 = u3 * eye
    if not minimal:
        h11 = h11 - (u1 - u2) / (2 * g1**2) * (vpid @ vpi)
        h22 = h22 - (u3 - u2) / (2 * g1**2) * (vpi @ vpid)

    return np.block([[h11, x], [x.conj().T, h22]])


def h_six_band(B, du, p: ABCParams, n_ll=256, valley=1, minimal=False):
    """Full six-band Hamiltonian in basis (A1, B1, A2, B2, A3, B3).

    ``minimal=True`` keeps only gamma0 and gamma1 (useful as a controlled
    check that the two-band projection is doing what it should).
    """
    p_ = p.minimal() if minimal else p
    omega = float(p.omega(B))
    vpi, vpid = pi_matrices(n_ll, valley, omega)
    eye = np.eye(n_ll)
    zero = np.zeros((n_ll, n_ll))
    u1, u2, u3 = layer_potentials(du)
    d = p_.delta

    v3pi = (p_.gamma3 / p_.gamma0) * vpi
    v3pid = (p_.gamma3 / p_.gamma0) * vpid
    v4pid = (p_.gamma4 / p_.gamma0) * vpid
    g1 = p_.gamma1 * eye
    g2 = 0.5 * p_.gamma2 * eye

    #            A1        B1          A2          B2          A3          B3
    rows = [
        [u1 * eye, vpid, -v4pid, v3pi, zero, g2],
        [None, (u1 + d) * eye, g1, -v4pid, zero, zero],
        [None, None, (u2 + d) * eye, vpid, -v4pid, v3pi],
        [None, None, None, (u2 + d) * eye, g1, -v4pid],
        [None, None, None, None, (u3 + d) * eye, vpid],
        [None, None, None, None, None, u3 * eye],
    ]
    for i in range(6):
        for j in range(i):
            rows[i][j] = rows[j][i].conj().T
    return np.block(rows)


def h_two_band_folded(B, du, p: ABCParams, n_ll=256, valley=1, minimal=False,
                      e_fold=0.0):
    """Exact Loewdin folding of the six-band model onto the (A1, B3) subspace.

    ``H_eff(E) = H_LL + H_LH (E - H_HH)^{-1} H_HL``, evaluated at ``E = e_fold``
    (default: charge neutrality).  This is exact for states at ``e_fold`` and
    is the honest "two-band model" in a magnetic field: unlike the analytic
    models above it is derived *after* minimal substitution, so it carries all
    the ladder-operator commutators that the B=0 Schrieffer-Wolff expansion
    silently drops.

    Comparing ``two_band`` against this isolates the error of the ``pi ->
    ladder`` substitution; comparing this against ``six_band`` isolates the
    error of restricting to two bands at all.
    """
    h = h_six_band(B, du, p, n_ll=n_ll, valley=valley, minimal=minimal)
    n = n_ll
    lo = np.r_[0:n, 5 * n:6 * n]          # A1, B3
    hi = np.r_[n:5 * n]                   # B1, A2, B2, A3  (the dimer sites)
    h_ll = h[np.ix_(lo, lo)]
    h_lh = h[np.ix_(lo, hi)]
    h_hh = h[np.ix_(hi, hi)]
    g = np.linalg.solve(e_fold * np.eye(hi.size) - h_hh, h_lh.conj().T)
    h_eff = h_ll + h_lh @ g
    return 0.5 * (h_eff + h_eff.conj().T)   # symmetrise away round-off


def build(model: str, B, du, p: ABCParams, n_ll=256, valley=1):
    """Dispatch to the requested model's Landau-level Hamiltonian."""
    if model == "two_band_minimal":
        return h_two_band(B, du, p, n_ll, valley, minimal=True)
    if model == "two_band":
        return h_two_band(B, du, p, n_ll, valley, minimal=False)
    if model == "two_band_folded":
        return h_two_band_folded(B, du, p, n_ll, valley, minimal=False)
    if model == "two_band_folded_minimal":
        return h_two_band_folded(B, du, p, n_ll, valley, minimal=True)
    if model == "six_band_minimal":
        return h_six_band(B, du, p, n_ll, valley, minimal=True)
    if model == "six_band":
        return h_six_band(B, du, p, n_ll, valley, minimal=False)
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


# ------------------------------------------------------------------ solver
def _to_ll_major(h: np.ndarray, nb: int) -> np.ndarray:
    """Reorder a sublattice-major matrix to Landau-index-major.

    In sublattice-major order the index is ``s*n_ll + n``; we want ``n*nb + s``
    so that the ladder couplings (which change ``n`` by a few) sit near the
    diagonal and the matrix becomes banded.
    """
    n_ll = h.shape[0] // nb
    perm = (np.arange(nb * n_ll).reshape(nb, n_ll).T).ravel()
    return h[np.ix_(perm, perm)]


def _banded_upper(h: np.ndarray, bw: int) -> np.ndarray:
    """Pack the upper triangle of a banded matrix into scipy's ``ab`` form."""
    n = h.shape[0]
    ab = np.zeros((bw + 1, n), dtype=h.dtype)
    for k in range(bw + 1):
        ab[bw - k, k:] = np.diag(h, k)
    return ab


def _bandwidth(model: str, nb: int) -> int:
    """Max |Delta n| coupling in the model, converted to a matrix bandwidth."""
    # six-band: only I, a, a^dg appear -> |Delta n| <= 1
    # two-band: a^3 appears            -> |Delta n| <= 3
    dn = 1 if nb == 6 else 3
    return dn * nb + (nb - 1)


#: Which layer each sublattice component of a model belongs to (0-indexed:
#: layer 0 is the bottom, layer 2 the top).  In the six-band basis
#: (A1,B1,A2,B2,A3,B3) that is simply ``s // 2``; the two-band models keep
#: only A1 (bottom) and B3 (top).
LAYER_OF = {
    "two_band_minimal": (0, 2),
    "two_band": (0, 2),
    "two_band_folded": (0, 2),
    "two_band_folded_minimal": (0, 2),
    "six_band_minimal": (0, 0, 1, 1, 2, 2),
    "six_band": (0, 0, 1, 1, 2, 2),
}

#: Index of the top layer.  With the linear potential drop of
#: `params.layer_potentials`, the top layer sits at ``u3 = +du/2``.
TOP_LAYER = 2


@dataclass
class Levels:
    """Landau levels for one (B, du, valley) point."""

    energy: np.ndarray          # eV, sorted
    B: float
    du: float
    valley: int
    model: str
    n_ll: int
    n_dropped: int = 0          # states rejected as truncation artefacts
    #: (n_levels, 3) fraction of each level's weight on layers 1, 2, 3.
    #: Rows sum to 1.  ``None`` unless the solve requested it.
    layer_weight: np.ndarray | None = None

    def __len__(self):
        return self.energy.size

    @property
    def top_weight(self):
        """Fraction of each level residing on the top layer."""
        if self.layer_weight is None:
            raise ValueError("solve with layers=True to get layer weights")
        return self.layer_weight[:, TOP_LAYER]


def _layer_weights(vecs: np.ndarray, model: str, n_ll: int) -> np.ndarray:
    """Per-level weight on each of the three layers, from LL-major eigenvectors.

    ``vecs`` is ``(nb*n_ll, n_levels)`` indexed as ``n*nb + s``, so reshaping
    to ``(n_ll, nb, n_levels)`` and summing |.|^2 over the Landau index gives
    the weight per sublattice, which then groups into layers.
    """
    nb = NBANDS[model]
    w = np.abs(vecs.reshape(n_ll, nb, -1)) ** 2
    per_sub = w.sum(axis=0)                       # (nb, n_levels)
    out = np.zeros((3, per_sub.shape[1]))
    for s, layer in enumerate(LAYER_OF[model]):
        out[layer] += per_sub[s]
    return (out / out.sum(axis=0, keepdims=True)).T   # (n_levels, 3)


def _as_range(window, e_range):
    """Normalise ``window`` / ``e_range`` into an ``(lo, hi)`` pair."""
    if e_range is not None:
        lo, hi = float(e_range[0]), float(e_range[1])
        if hi <= lo:
            raise ValueError(f"e_range must be increasing, got {e_range}")
        return lo, hi
    return -float(window), float(window)


def solve_levels_vals(
    model: str,
    B: float,
    du: float,
    p: ABCParams,
    n_ll: int = 256,
    valley: int = 1,
    window: float = 0.3,
    e_range=None,
) -> np.ndarray:
    """Landau-level energies only -- no eigenvectors, no truncation filter.

    Roughly 20x faster than :func:`solve_levels`, because the eigenvector step
    of the banded solver dominates.  It carries no truncation check of its
    own, so it is only safe behind :func:`solve_auto`, which compares two
    basis sizes instead.
    """
    nb = NBANDS[model]
    lo, hi = _as_range(window, e_range)
    h = build(model, B, du, p, n_ll=n_ll, valley=valley)
    h = _to_ll_major(h, nb)
    if model in DENSE_MODELS:
        vals = np.linalg.eigvalsh(h)
        return np.sort(vals[(vals >= lo) & (vals <= hi)])
    ab = _banded_upper(h, _bandwidth(model, nb))
    return np.sort(eig_banded(ab, lower=False, select="v",
                              select_range=(lo, hi), eigvals_only=True))


def solve_levels(
    model: str,
    B: float,
    du: float,
    p: ABCParams,
    n_ll: int = 256,
    valley: int = 1,
    window: float = 0.3,
    tail_frac: float = 0.15,
    tail_tol: float = 1e-6,
    e_range=None,
    layers: bool = False,
) -> Levels:
    """Landau levels inside the energy window for one (B, du, valley).

    Uses a banded eigensolver on the Landau-index-major reordering, selecting
    only the energy window of interest, then filters out states contaminated by
    the basis truncation.  Computes eigenvectors, so it is ~20x more expensive
    than :func:`solve_levels_vals`.
    """
    nb = NBANDS[model]
    lo, hi = _as_range(window, e_range)
    h = build(model, B, du, p, n_ll=n_ll, valley=valley)
    h = _to_ll_major(h, nb)

    if model in DENSE_MODELS:
        vals, vecs = np.linalg.eigh(h)
        sel = (vals >= lo) & (vals <= hi)
        vals, vecs = vals[sel], vecs[:, sel]
    else:
        bw = _bandwidth(model, nb)
        ab = _banded_upper(h, bw)
        vals, vecs = eig_banded(
            ab, lower=False, select="v", select_range=(lo, hi)
        )

    # weight in the top tail of Landau indices -> truncation artefact
    n_tail = max(1, int(round(tail_frac * n_ll)))
    tail_start = (n_ll - n_tail) * nb
    tail_weight = np.sum(np.abs(vecs[tail_start:, :]) ** 2, axis=0)
    keep = tail_weight <= tail_tol

    vals, vecs = vals[keep], vecs[:, keep]
    order = np.argsort(vals)
    lw = _layer_weights(vecs[:, order], model, n_ll) if layers else None

    return Levels(
        energy=vals[order],
        B=float(B),
        du=float(du),
        valley=int(valley),
        model=model,
        n_ll=n_ll,
        n_dropped=int(np.sum(~keep)),
        layer_weight=lw,
    )


def suggest_n_ll(model: str, B: float, p: ABCParams, window: float = 0.3,
                 safety: float = 6.0, n_min: int = 96, n_max: int = 4096) -> int:
    """Heuristic *starting* basis size; always refine with :func:`solve_auto`.

    Two-band:  E_n ~ omega^3 n^{3/2} / gamma1^2  ->  n ~ (window gamma1^2 /
    omega^3)^{2/3}.  Six-band: the highest relevant branch is Dirac-like,
    E_n ~ omega sqrt(n)  ->  n ~ (window / omega)^2.

    The safety factor is deliberately large.  The pure-cubic estimate badly
    undercounts the true number of levels once gamma2 is on: it splits the
    cubic touching into three Dirac cones with a small velocity, and the
    closely spaced levels around them can easily double or triple the level
    count inside a given window.
    """
    omega = float(p.omega(B))
    n_cubic = (window * p.gamma1**2 / omega**3) ** (2.0 / 3.0)
    n_dirac = (window / omega) ** 2
    n = safety * max(n_cubic, n_dirac)
    return int(np.clip(np.ceil(n), n_min, n_max))


#: a converged solve still rejects a few states whose tails genuinely reach the
#: top of the basis; more than this many means the basis is too small.
DROP_FLOOR = 6

#: Cutting the ladder leaves the ABC dimer chain with dangling links, which
#: produce a few spurious SSH-like states near E = 0.  Empirically 3 per
#: valley for the six-band model.  More than this many unmatched levels means
#: the basis is genuinely too small, not merely truncated.
ARTIFACT_CAP = 8

#: When the fast path sees *zero* unmatched levels, its artefact detector is
#: blind and it must hand over to the eigenvector path -- see `solve_auto`.
#:
#: Cutting the ladder always leaves dangling links, so a truncated solve
#: always carries a few spurious states.  In the full six-band model gamma2
#: pushes them off zero and they drift by ~1e-4 eV between basis sizes, where
#: the matching catches them (genuine levels agree to ~1e-16).  But with
#: gamma2 off -- `six_band_minimal` -- or in the two-band models, where the
#: truncated ``a^3`` leaves states with no partner at all, the artefacts are
#: *exact* zero modes: pinned to E = 0 in every basis size and
#: indistinguishable from the genuine zero-mode multiplet by energy alone.
#: Finding none is therefore the signature of exactly that blindness, not of
#: a clean solve.
#:
#: That fallback catches total blindness but not *partial* blindness (one
#: artefact drifting while another stays pinned, which `six_band_minimal`
#: does at finite dU).  So the fast path is additionally restricted to the
#: models where `test_fast_solver_matches_eigenvector_solver` demonstrates
#: exact agreement -- which is the production model, the only one the DOS
#: maps use.  Everything else takes the eigenvector path, which is cheap for
#: a two-band model anyway (3x smaller in each dimension).
FAST_SAFE = frozenset({"six_band"})


def _common(b, a, tol):
    """Elements of sorted ``b`` that have a partner in sorted ``a`` within ``tol``."""
    if a.size == 0 or b.size == 0:
        return np.empty(0)
    i = np.clip(np.searchsorted(a, b), 1, a.size - 1)
    near = np.minimum(np.abs(b - a[i - 1]), np.abs(b - a[i]))
    return b[near <= tol]


def solve_auto(
    model: str,
    B: float,
    du: float,
    p: ABCParams,
    valley: int = 1,
    window: float = 0.3,
    n_ll: int | None = None,
    max_doublings: int = 5,
    atol: float = 1e-9,
    strict: bool = True,
    verify: bool = False,
    e_range=None,
    fast: bool | None = None,
    match_tol: float = 1e-11,
    layers: bool = False,
) -> Levels:
    """Solve, growing the basis until no genuine levels are being discarded.

    This is the entry point to use for production runs.  Plain
    :func:`solve_levels` with a hand-picked ``n_ll`` can silently discard
    genuine levels as truncation artefacts, which shifts every branch index
    after them and corrupts anything built on top (filling, DOS, fan plots).

    The number of rejected states is the convergence signal: a converged solve
    rejects a small constant (``DROP_FLOOR``) of genuine edge states, whereas
    an undersized basis rejects tens.  ``verify=True`` additionally confirms
    the result against a doubled basis, which is ~10x more expensive and is
    meant for spot checks rather than production grids.

    Raises ``RuntimeError`` if convergence is not reached and ``strict``.
    """
    lo, hi = _as_range(window, e_range)
    w_size = max(abs(lo), abs(hi))
    n = n_ll if n_ll is not None else suggest_n_ll(model, B, p, w_size)
    if fast is None:
        fast = model in FAST_SAFE
    if layers:
        # layer weights come from the eigenvectors, which the fast
        # eigenvalue-only path never forms
        fast = False

    if fast:
        # Two eigenvalue-only solves at different basis sizes, keeping only
        # the levels common to both.  Costs far less than one eigenvector
        # solve.
        #
        # The separation is enormous but `match_tol` still has to be chosen
        # with care.  Genuine levels agree to 0 or ~1e-16 eV (pure roundoff).
        # Artefacts typically move by ~1e-3 eV -- but not always: one that
        # lands near a flat valley-polarised zero mode is pinned by the same
        # mechanism as the real level and was measured drifting only 2e-9 eV.
        # So the tolerance sits at 1e-11: ~100x above worst-case roundoff for
        # a large basis, ~200x below the stickiest artefact observed.  At
        # 1e-7 that artefact survived and put a spurious horizontal stripe
        # across the DOS map.
        a = solve_levels_vals(model, B, du, p, n_ll=n, valley=valley,
                              e_range=(lo, hi))
        for _ in range(max_doublings):
            m = int(np.ceil(1.5 * n))
            b = solve_levels_vals(model, B, du, p, n_ll=m, valley=valley,
                                  e_range=(lo, hi))
            keep = _common(b, a, match_tol)
            n_art = len(b) - keep.size
            if n_art == 0:
                fast = False        # detector is blind here; hand over below
                break
            if n_art <= ARTIFACT_CAP:
                return Levels(energy=keep, B=float(B), du=float(du),
                              valley=int(valley), model=model, n_ll=m,
                              n_dropped=n_art)
            n, a = 2 * n, solve_levels_vals(model, B, du, p, n_ll=2 * n,
                                            valley=valley, e_range=(lo, hi))
        else:
            if strict:
                raise RuntimeError(
                    f"{model} at B={B} T, du={du} eV, valley={valley}: "
                    f"{n_art} levels still unmatched between basis sizes "
                    f"after {max_doublings} doublings (n_ll={n}); expected at "
                    f"most {ARTIFACT_CAP} truncation artefacts. Narrow the "
                    f"energy range or raise `max_doublings`.")
            return Levels(energy=keep, B=float(B), du=float(du),
                          valley=int(valley), model=model, n_ll=n,
                          n_dropped=n_art)

    cur = solve_levels(model, B, du, p, n_ll=n, valley=valley,
                       e_range=(lo, hi), layers=layers)
    for _ in range(max_doublings):
        if cur.n_dropped <= DROP_FLOOR:
            break
        n *= 2
        cur = solve_levels(model, B, du, p, n_ll=n, valley=valley,
                           e_range=(lo, hi), layers=layers)
    else:
        if strict:
            raise RuntimeError(
                f"{model} at B={B} T, du={du} eV, valley={valley}: basis not "
                f"converged after {max_doublings} doublings (n_ll={n}, "
                f"{cur.n_dropped} states rejected). Narrow `window` or raise "
                f"`max_doublings`.")

    if verify:
        # expensive: confirm the level set is unchanged in a doubled basis.
        chk = solve_levels(model, B, du, p, n_ll=2 * n, valley=valley,
                           e_range=(lo, hi), layers=False)
        if len(chk) != len(cur) or not np.allclose(chk.energy, cur.energy,
                                                   atol=atol):
            raise RuntimeError(
                f"{model} at B={B} T, du={du} eV, valley={valley}: level set "
                f"changed on doubling the basis ({len(cur)} -> {len(chk)} "
                f"levels). The drop-count heuristic missed this.")
    return cur


def converged_levels(
    model: str,
    B: float,
    du: float,
    p: ABCParams,
    valley: int = 1,
    window: float = 0.3,
    n_ll: int | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_doublings: int = 4,
    verbose: bool = False,
):
    """Solve, doubling the basis until the retained levels stop moving.

    Returns ``(Levels, max_abs_shift)``.
    """
    n = n_ll if n_ll is not None else suggest_n_ll(model, B, p, window)
    prev = solve_levels(model, B, du, p, n_ll=n, valley=valley, window=window)
    for _ in range(max_doublings):
        n *= 2
        cur = solve_levels(model, B, du, p, n_ll=n, valley=valley, window=window)
        m = min(len(prev), len(cur))
        if m == 0:
            return cur, 0.0
        shift = float(np.max(np.abs(cur.energy[:m] - prev.energy[:m])))
        same_count = len(prev) == len(cur)
        if verbose:
            print(f"  n_ll={n:5d}  n_levels={len(cur):4d}  max shift={shift:.3e}")
        if same_count and shift <= atol + rtol * max(1e-12, window):
            return cur, shift
        prev = cur
    return prev, shift


def spectrum_vs(
    model: str,
    p: ABCParams,
    B_values,
    du_values,
    valleys=(+1, -1),
    window: float = 0.3,
    n_ll=None,
):
    """Landau levels on a (B, du) grid.

    Returns a dict ``(iB, idu, valley) -> energy array``.  Basis size is chosen
    per B unless ``n_ll`` is given explicitly.
    """
    B_values = np.atleast_1d(B_values)
    du_values = np.atleast_1d(du_values)
    out = {}
    for ib, B in enumerate(B_values):
        n = n_ll if n_ll is not None else suggest_n_ll(model, B, p, window)
        for iu, du in enumerate(du_values):
            for xi in valleys:
                lv = solve_levels(model, B, du, p, n_ll=n, valley=xi, window=window)
                out[(ib, iu, xi)] = lv.energy
    return out
