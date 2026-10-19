"""Semiclassical Onsager Landau-level fan for ABC trilayer graphene.

Feeds an ABC band structure to the orbit-area / Onsager-quantization engine in
`hofstadter/source_code/semiclassical`, which is otherwise driven by a moire
bilayer.  Nothing in that tree is modified: `abc_ll.semiclassical.engine()`
imports `isoenergy.get_energy_resolved_data` and `onsager.onsager_fan_band` and
this script calls them directly, which is the API documented under
"Post-processing pipeline" in its user guide.

Two departures from the engine's defaults, both deliberate:

**Bmultiplier = 1, not 4.**  The engine's 4 is an untraced empirical factor tied
to its Hofstadter flux convention (`doc_technical.md`, "The Bmultiplier
factor").  There is no moire cell and no flux convention here, so the textbook
Onsager condition applies.  Checked analytically -- see the module docstring of
`tests/test_semiclassical.py`.

**chi_factor = 0.**  No Fukuyama susceptibility term; the `_SBMC` level is not
produced.  `_S` (area only), `_SB` (+ Berry phase) and `_SBM` (+ orbital
moment) all come out of a single run because the engine accumulates them.

**Read `_SBM`, not `_SB`.**  Scored against the exact levels from
`abc_ll.landau` at B = 10 T, as a fraction of the exact level spacing:

    model             dU (meV)     S      SB     SBM
    two_band_minimal    10       0.373   0.118   0.098
    two_band_minimal   100       0.532   0.198   0.142
    six_band            20       0.472   0.137   0.070
    six_band           100       0.303   0.199   0.070

The Berry phase and the orbital moment are not alternatives -- both are first
order in B and both are needed.  `_SB` alone is better than bare Onsager but
leaves about twice the error of `_SBM`.  (At six_band dU = 200 meV `_S` scores
0.081 against `_SB` 0.324; that is a coincidence of one point, and `_SBM` is
0.085 there.)  `_SB` is kept because the Berry term is what was asked for and
because the pair bracket the size of each correction.

**Gapped displacement fields only.**  At dU = 0 the band touching is gapless and
the Berry curvature collapses to a delta function at the touching points, which
no grid sum can see: `enclosedBC` comes out zero and `_SB` degenerates to `_S`.
Every dU is still computed, but `flux_converged` flags the ones where the
enclosed flux has not saturated near its topological value, and those should not
be read as Berry-corrected.

Run:  python3 scripts/export_semiclassical.py six_band
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import matio, semiclassical as sc  # noqa: E402
from abc_ll.dos import moire_gate  # noqa: E402
from abc_ll.params import DEFAULT as _P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

_ALIAS = {"6band": "six_band", "6band_minimal": "six_band_minimal",
          "2band": "two_band_minimal", "2band_minimal": "two_band_minimal"}
_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "six_band")
MODEL = _ALIAS.get(_arg, _arg)
# multiprocessing on Windows spawns rather than forks, and the child does not
# inherit sys.argv -- it would silently fall back to the default model.  Put
# the choice in the environment, which the child does inherit.
os.environ["MODEL"] = MODEL
P = _P.minimal() if MODEL.endswith("minimal") else _P

NDU = int(os.environ.get("NDU", 41))
DU_MAX = float(os.environ.get("DU_MAX", 0.20))
BMIN = float(os.environ.get("BMIN", 1.0))
BMAX = float(os.environ.get("BMAX", 12.0))
NB = int(os.environ.get("NB", 45))
NMAX = int(os.environ.get("NMAX", 120))        # max Landau index
NK = int(os.environ.get("NK", 376))            # k-mesh side; at this spacing
                                               # doubling nk moves the enclosed
                                               # flux by < 1e-4
KMAX_EV = float(os.environ.get("KMAX_EV", 0.50))   # patch half-width, hbar*v*k.
                                               # The worst case in the dU grid
                                               # -- the 160 meV orbit at
                                               # dU = 200 meV -- reaches 0.4002,
                                               # so 0.40 loses it to the border.
NE = int(os.environ.get("NE", 400))            # energy samples per band
E_WIN = float(os.environ.get("E_WIN", 0.16))   # eV, matches the LL exports
ETA = float(os.environ.get("ETA", 2e-3))       # eV, Kubo level repulsion
KT = float(os.environ.get("KT", 3.0))          # meV, smearing for dL/dE
NPOCKET = int(os.environ.get("NPOCKET", 2))    # orbits quantized per band
NPROC = int(os.environ.get("NPROC", 0))        # 0 = serial

BMULTIPLIER = 1.0
TERM_FACTORS = (1.0, 1.0, 0.0)                 # BC, orbital moment, chi
LIFSHITZ = float(os.environ.get("LIFSHITZ", 50))

#: Total Berry flux of a band over the whole patch.  For a gapped chirality-3
#: band this is the topological +-3*pi, and it is the right test of whether the
#: curvature is present and resolved at all.  It is NOT the same as the flux
#: *enclosed by an orbit*, which is legitimately smaller whenever the orbit is
#: no bigger than the region the curvature occupies -- that is the energy
#: dependence the Berry correction exists to capture, not an error.
#: The tolerance is deliberately loose: this is a gapless detector, separating
#: ~3*pi from ~0, not a convergence test.  six_band runs up to 3.5*pi at
#: dU = 200 meV because the gamma1-split remote bands contribute curvature too,
#: which is physical.
FLUX_TARGET = 3.0 * np.pi
FLUX_TOL = float(os.environ.get("FLUX_TOL", 0.5))


def bc_sign(area_col, E_levels):
    """Sign of the Berry term for one orbit: -1 electron-like, +1 hole-like.

    The engine applies a single fixed `BC_factor` to every orbit, on the
    argument that the sign comes from the carrier charge and the field
    direction rather than from the orbit (`doc_technical.md`, "Term signs").
    That holds within one band but not across a conduction/valence pair, where
    two things flip at once: the Berry curvature itself is opposite in the two
    bands, and the sense in which the orbit is traversed reverses, because
    hbar*kdot = -e v x B runs k perpendicular to v = grad E and grad E points
    outward on an electron pocket but inward on a hole pocket.  The physical
    Onsager phase is the same for both; the stored `enclosedBC` is not, so the
    prefactor has to undo it.

    Leaving it fixed is what made the conduction band of every model look
    systematically shifted while the valence band fitted well.  Measured on
    two_band_minimal at B = 9 T, median |dE| over the upper two thirds of the
    levels, in meV:

        dU (meV)   cond +1   cond -1    val +1   val -1
            40       6.717     2.135     1.999    8.305
            60       3.608     2.096     1.981    4.918
           100       3.515     1.883     1.866    3.141
           160       3.637     1.616     1.630    4.612

    The winning columns agree to three digits, as they must: that model is
    particle-hole symmetric, so a correct prescription has to give the two
    bands the same spectrum.  The orbital-moment term is *not* flipped -- Lz
    is identical in the two bands of a two-band model and the measured fan
    confirms it (flipping it drops a level and moves the rest further off).

    Keying on `dS/dE` rather than on `sign(Phi_B)` is deliberate: it is orbit
    character that the traversal argument depends on, so this also gets the
    inner orbit of a Mexican-hat band right, where the area shrinks with
    energy inside the same band that is electron-like further out.
    """
    m = area_col > 0
    if m.sum() < 3:
        return 1.0
    slope = np.polyfit(E_levels[m], area_col[m], 1)[0]
    return -1.0 if slope > 0 else 1.0


def _band_fans(bs, Blist, valley, iband):
    """Fans and diagnostics for one band of one valley, pocket by pocket."""
    get_energy_resolved_data, onsager_fan_band, _ = sc.engine()

    E = bs[f"E_{valley}"][iband]
    Oz = bs[f"Oz_{valley}"][iband]
    Lz = bs[f"Lz_{valley}"][iband]

    # The engine's own run_isoenergy uses each band's full range; we clip to the
    # window so the nE samples are not spent far outside it.
    emin = max(E.min(), -E_WIN * 1e3)
    emax = min(E.max(), E_WIN * 1e3)
    if emax <= emin:
        return {}, np.nan, np.array([])

    E_levels = np.linspace(emin, emax, NE)
    area, enclosedBC, dL_dE = get_energy_resolved_data(
        KT, E, Oz, Lz, E_levels, bs["vol_M"], bs["nk1"], bs["nk2"], False)

    # Informational: flux enclosed by the largest orbit at the top of the
    # window.  Below 3*pi whenever the orbit is no bigger than the curvature
    # region, which is physics, not error -- the convergence flag uses the
    # total patch flux instead.
    have = np.flatnonzero(area[:, 0] > 0)
    flux_top = enclosedBC[have[-1], 0] if have.size else np.nan

    fans = {}
    signs = {}
    for p in range(min(NPOCKET, area.shape[1])):
        if not np.any(area[:, p] > 0):
            continue
        bcf = bc_sign(area[:, p], E_levels)
        ll = onsager_fan_band(
            Blist, NMAX, E_levels, area[:, [p]], enclosedBC[:, [p]], dL_dE,
            dChi_dE=None, term_factors=(bcf,) + TERM_FACTORS[1:],
            Bmultiplier=BMULTIPLIER, lifshitz_threshold=LIFSHITZ)
        if ll:
            signs[p] = bcf
            for suffix, LL in ll.items():
                fans[f"p{p}_{suffix}"] = LL
    return fans, flux_top, signs


def _one_du(args):
    """Everything for a single displacement field.  Picklable for Pool.map."""
    du, Blist = args
    bs = sc.band_structure(MODEL, du, P, kmax_ev=KMAX_EV, nk=NK, eta=ETA,
                           nband_keep=2)
    nk_tot = bs["E_K"].shape[1]
    kweight = (2.0 * np.pi) ** 2 / (nk_tot * bs["vol_M"])

    out = {"fans": {}, "flux": {}, "total": {}, "bcsign": {}}
    for valley in ("K", "Kp"):
        for iband in range(bs[f"E_{valley}"].shape[0]):
            fans, flux, signs = _band_fans(bs, Blist, valley, iband)
            for key, LL in fans.items():
                out["fans"][f"{valley}_b{iband}_{key}"] = LL
            for p, s in signs.items():
                out["bcsign"][f"{valley}_b{iband}_p{p}"] = s
            out["flux"][f"{valley}_b{iband}"] = flux
            out["total"][f"{valley}_b{iband}"] = (
                bs[f"Oz_{valley}"][iband].sum() * kweight)
    return out


def main():
    du = np.linspace(-DU_MAX, DU_MAX, NDU)
    Blist = np.linspace(BMIN, BMAX, NB)

    sc.engine()   # fail early and loudly if the engine is not reachable
    print(f"{MODEL}: semiclassical Onsager fan")
    print(f"  {NDU} dU from {du[0]*1e3:.0f} to {du[-1]*1e3:.0f} meV, "
          f"{NB} B from {BMIN} to {BMAX} T, nmax = {NMAX}")
    print(f"  k patch +-{KMAX_EV} eV at nk = {NK}, nE = {NE}, eta = {ETA*1e3} meV")
    print(f"  Bmultiplier = {BMULTIPLIER}, term_factors = {TERM_FACTORS} "
          f"(BC on, chi off)")

    t0 = time.perf_counter()
    args = [(float(u), Blist) for u in du]
    if NPROC > 1:
        import multiprocessing
        with multiprocessing.Pool(NPROC) as pool:
            results = []
            for i, r in enumerate(pool.imap(_one_du, args)):
                results.append(r)
                print(f"\r  {100*(i+1)//NDU}%", end="", flush=True)
        print()
    else:
        results = []
        for i, a in enumerate(args):
            results.append(_one_du(a))
            print(f"\r  {100*(i+1)//NDU}%", end="", flush=True)
        print()
    print(f"  {time.perf_counter()-t0:.0f} s")

    # Collect into (nDU, nB, nmax+1) arrays.  Keys vary with dU -- a pocket may
    # exist only at some fields, and the engine adds _seg suffixes where the
    # area curve has Lifshitz jumps -- so allocate from the union and leave NaN
    # where a dU did not produce that key.
    all_keys = sorted({k for r in results for k in r["fans"]})
    out = {}
    for key in all_keys:
        arr = np.full((NDU, NB, NMAX + 1), np.nan)
        for i, r in enumerate(results):
            if key in r["fans"]:
                arr[i] = r["fans"][key]
        out[f"LL_{key}"] = arr

    bc_keys = sorted({k for r in results for k in r["bcsign"]})
    bcsign = np.full((NDU, len(bc_keys)), np.nan)
    for j, key in enumerate(bc_keys):
        for i, r in enumerate(results):
            bcsign[i, j] = r["bcsign"].get(key, np.nan)

    flux_keys = sorted({k for r in results for k in r["flux"]})
    flux = np.full((NDU, len(flux_keys)), np.nan)
    total = np.full((NDU, len(flux_keys)), np.nan)
    for j, key in enumerate(flux_keys):
        for i, r in enumerate(results):
            flux[i, j] = r["flux"].get(key, np.nan)
            total[i, j] = r["total"].get(key, np.nan)
    converged = np.abs(np.abs(total) - FLUX_TARGET) < FLUX_TOL * FLUX_TARGET

    nbad = int((~converged).any(axis=1).sum())
    print(f"  total band Berry flux is +-3 pi at {NDU - nbad}/{NDU} dU values "
          f"(tol {FLUX_TOL:.0%})")
    if nbad:
        bad = du[(~converged).any(axis=1)] * 1e3
        print("    off target at dU (meV): "
              + ", ".join(f"{b:.0f}" for b in bad))
        print("    -- gapless there, so the curvature is a delta function the "
              "grid cannot see; _SB degenerates to _S and must not be read as "
              "Berry-corrected")

    gate = moire_gate()
    out.update({
        "model": MODEL,
        "dU_eV": du, "dU_meV": du * 1e3,
        "Blist": Blist, "nmax": NMAX,
        "onsager_Bmultiplier": BMULTIPLIER,
        "term_factors": np.array(TERM_FACTORS),
        "flux_keys": np.array(flux_keys, dtype=object),
        "bc_keys": np.array(bc_keys, dtype=object),
        "bc_sign": bcsign,
        "enclosed_flux_top": flux,
        "enclosed_flux_top_over_pi": flux / np.pi,
        "total_flux": total,
        "total_flux_over_pi": total / np.pi,
        "flux_converged": converged.astype(float),
        "nk": NK, "kmax_ev": KMAX_EV, "nE": NE, "eta_eV": ETA, "kT_meV": KT,
        "energy_window_eV": E_WIN,
        "n_m2": gate.density(du), "n_cm2": gate.density(du) / 1e4,
        "filling_moire": 4.0 * du / gate.du_max,
        "params": matio.param_struct(P),
        "gate": matio.gate_struct(gate),
        "units": {
            "LL_*": "meV, (nDU, nB, nmax+1).  Key is "
                    "LL_<valley>_b<band>_p<pocket>_<terms>, band 0 = valence "
                    "and 1 = conduction of the low-energy pair, pocket 0 = "
                    "largest orbit.  terms: S = orbit area only, SB = + Berry "
                    "phase, SBM = + orbital moment.  _segN appears where the "
                    "area curve has a Lifshitz jump and each monotonic branch "
                    "is solved separately.  NaN where no root exists.",
            "enclosed_flux_top": "Berry flux enclosed by the largest orbit at "
                                 "the top of the window, per flux_keys.  Less "
                                 "than 3*pi when the orbit is no bigger than "
                                 "the curvature region -- expected, not error.",
            "total_flux": "Berry flux of the whole band over the patch, per "
                          "flux_keys; +-3*pi for a gapped chirality-3 band, "
                          "zero when gapless.  This is what flux_converged "
                          "tests.",
            "bc_sign": "(nDU, n_bc_keys) the Berry prefactor used for each "
                       "orbit, per bc_keys: -1 on electron-like orbits and "
                       "+1 on hole-like ones.  The engine applies one fixed "
                       "sign; across a conduction/valence pair both the "
                       "curvature and the traversal sense flip, so it has to "
                       "be per orbit.  See bc_sign() in the exporter.",
            "Blist": "tesla",
            "dU_eV": "eV, interlayer potential difference u3 - u1",
        },
        "readme": (
            "Semiclassical Onsager Landau levels for ABC trilayer graphene, "
            "computed with the orbit-area engine from "
            "hofstadter/source_code/semiclassical driven by an ABC band "
            "structure.  Onsager condition: S(E)/(2*pi)^2 = (|B|/phi0) * "
            "[(n+1/2) - sign(B)*Phi_B/(2*pi) - (dL/dE)/(2*pi)], with "
            "Bmultiplier = 1 (the textbook value; the engine's default of 4 is "
            "a Hofstadter-specific compensator that does not apply here) and "
            "no susceptibility term.  Per valley and spinless: a level holds "
            "e*B/h states.  The Berry term needs a gapped band structure -- "
            "see flux_converged."),
    })

    path = os.path.join(CACHE, f"semiclassical_{MODEL}.mat")
    savemat(path, out, do_compression=True)
    print(f"  wrote {path}  ({len(all_keys)} fan arrays)")


if __name__ == "__main__":
    main()
