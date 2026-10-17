# ABC trilayer graphene: Landau levels in a displacement field

Landau-level spectra and density of states for ABC (rhombohedral) trilayer
graphene in a perpendicular magnetic field `B` and a vertical displacement
field, at several controlled levels of approximation.

```
abc_ll/params.py      parameters, constants, unit helpers
abc_ll/landau.py      Landau-level Hamiltonians + banded eigensolver
abc_ll/bands.py       zero-field band structures (validation + band edges)
abc_ll/dos.py         broadening, filling, chemical potential, DOS,
                      layer projection, moire gate calibration
abc_ll/plotstyle.py   plot style and continuous Landau-branch tracing
abc_ll/matio.py       self-describing .mat export for MATLAB
scripts/              compare_models.py, plot_spectra.py, dos_map.py,
                      export_levels.py
matlab/               MATLAB plotting scripts (base MATLAB, no toolboxes)
tests/                31 validation tests -- run these first
figures/              generated output
data/                 cached (B, dU) maps and .mat exports
```

Run `python3 tests/test_validation.py`, then `scripts/compare_models.py`
(model-error tables), `scripts/plot_spectra.py` (Landau fans) and
`scripts/dos_map.py` (the DOS map; `NB=.. NU=..` set the grid, results are
cached in `data/`) and `scripts/export_levels.py` (raw levels to .mat).

## Status

Working and tested: 33/33 in `tests/test_validation.py`.

The export pipeline writes the charge-neutrality energy (`cnp_eV`) itself,
so a run from a clean checkout needs no extra step:

```
python3 tests/test_validation.py      # 33 tests
python3 scripts/compare_models.py     # model-error tables
python3 scripts/plot_spectra.py       # Landau fans
python3 scripts/dos_map.py            # DOS map + .mat   (~35 min, cached)
python3 scripts/export_levels.py      # raw levels -> .mat (~40 min, cached)
```

`data/` and `figures/` are gitignored -- large (one export is 75 MB) and
fully reproducible from the above.

`scripts/add_cnp_to_mat.py` is a repair utility only.  It patches `cnp_eV`
into `.mat` files produced *before* that fix; files written by the current
code already carry it.

## Conventions

* Energies in **eV**, lengths in **m**, densities in **m^-2**.
* Sublattice basis `(A1, B1, A2, B2, A3, B3)`.  ABC stacking makes the dimer
  pairs `(B1, A2)` and `(B2, A3)`, so the low-energy sites are `A1` and `B3`,
  and `A1` sits directly below `B3` -- that pair carries `gamma2`.  (There is
  no `gamma5` in ABC: no A-A pair is vertically aligned across layers 1 and 3.)
* Valley `xi = +1` is K, `-1` is K'; `pi = xi*pi_x + i*pi_y`.
* Linear potential drop: `(u1, u2, u3) = (-dU/2, 0, +dU/2)`, with
  `dU = u3 - u1` the total interlayer potential difference.
* `E = 0` is charge neutrality, and `n` is measured from it.  The 12-fold
  zero-mode multiplet is therefore **half filled** at `n = 0`, spanning
  `nu = -6 .. +6`.

## The model hierarchy

| key | what it is |
|---|---|
| `two_band_minimal` | cubic band touching `(v pi^dg)^3/gamma1^2` + rigid `+-dU/2`.  The textbook model. |
| `two_band` | + gap renormalisation `-(u_i-u2) v^2 pi^dg pi/(2 gamma1^2)`, `gamma2/2`, and the gamma3 term `-(v v3/gamma1)(pi pi^dg + pi^dg pi)` |
| `two_band_folded` | exact Loewdin folding of the six-band model onto (A1, B3) at `E=0` |
| `six_band_minimal` | full 6x6, `gamma0` and `gamma1` only |
| `six_band` | full 6x6, `gamma0..gamma4` and `delta` |

`two_band_folded` exists to separate two error sources that are otherwise
confounded: it is derived *after* minimal substitution, so comparing it to
`two_band` isolates the cost of substituting `pi -> ladder` into a zero-field
effective Hamiltonian, while comparing it to `six_band_minimal` isolates the
cost of keeping only two bands.

A useful exact result falls out: **at `dU = 0` the analytic cubic model and
the folded model are identical to machine precision.**  The 4x4 dimer block of
the ABC chain has determinant `gamma1^4` with no momentum dependence, so the
third-order Schrieffer-Wolff cubic term is exact rather than perturbative.
The two diverge only once `dU != 0`, because the analytic model treats the
layer potential only to second order.

## Numerics

Everything is built as an explicit matrix in a truncated oscillator basis via
`a` and `a^dg`.  That is brute force, but it is cheap -- reordered to
Landau-index-major the Hamiltonian is banded (bandwidth 11 for six-band,
7 for two-band), so `scipy.linalg.eig_banded` with an energy-window selection
is fast -- and it removes all per-sublattice Landau-index bookkeeping, which
is what makes these calculations error-prone once `gamma2/gamma3/gamma4`
(which shift the index by different amounts) are switched on.

**Always use `landau.solve_auto`, not `solve_levels` with a hand-picked
`n_ll`.**  A truncated `a a^dg` is wrong in its last entry, so states with
weight near the top of the basis are spurious and get filtered out.  If the
basis is too small that filter starts eating *genuine* levels -- silently,
and in a B-dependent way, which shifts every branch index after them and
corrupts the filling, the DOS and the fan plots downstream.  This is a real
trap here: the natural `E_n ~ omega^3 n^{3/2}/gamma1^2` estimate undercounts
badly once `gamma2` is on, because `gamma2` splits the cubic touching into
three Dirac cones whose small velocity packs many extra levels into the same
window.  At `B = 1.5 T` the naive estimate lost ~45 of 103 real levels.
`solve_auto` grows the basis until the rejected-state count falls to its
floor; `verify=True` additionally confirms against a doubled basis.

## What the models are worth

From `scripts/compare_models.py`, errors in meV against the full six-band
model, on the 6 lowest conduction and 6 highest valence levels:

```
    B     dU      E1 | (a) ladder (b) 2-band (c) remote |   total
  1.0   0.00     6.2 |    0.00       1.00       5.40    |    5.79
  1.0   0.20    85.4 |    0.34      19.08      24.88    |   33.84
  4.0   0.10    49.1 |    0.77       5.89      15.52    |   16.65
 12.0   0.00     2.8 |    0.00      58.69      60.83    |   99.21
 12.0   0.20    82.8 |   12.76      26.88      24.77    |   37.10
```

* **(a)** the `pi -> ladder` substitution is a non-issue below ~4 T and stays
  under ~13 meV even at 12 T.  Refining the zero-field projection is not where
  the accuracy is.
* **(b)** restricting to two bands is the dominant error at high field.
* **(c)** the remote hoppings cost a roughly `dU`-independent 15-25 meV and are
  the *largest* single error at low field.

At zero field the extended two-band model is worth having at low energy
(0.97 meV vs 11.89 meV error below `hbar v k = 50 meV` at `dU = 0`), but by
`hbar v k = 0.25 eV` both two-band models are 60-90 meV off and the minimal
six-band model is the better approximation.

### Band-edge structure (six-band, zero field)

```
    dU |  gap (meV)   hat_c    hat_v  | 2b-min gap
  0.00 |       0.04    3.63    16.32  |      0.00
  0.05 |      39.68   -0.00    14.17  |     50.00
  0.10 |      78.60    1.90    21.48  |    100.00
  0.20 |     141.38   15.04    44.58  |    200.00
```

The gap is strongly **sublinear** in `dU` -- 141 meV at `dU = 200 meV`, where
the minimal two-band model says 200 meV, a 42% overestimate.  Both band edges
sit at finite `k` (Mexican hat), the valence one much more so (`hat_v` up to
45 meV), which puts a van Hove peak right at the valence edge.  Note the
minimal two-band model has **no hat at all**: `sqrt((dU/2)^2 + c^2 k^6)` is
monotonic.  The hat needs either the `k`-dependent diagonal or `gamma2`.

## The DOS stage

**Gate calibration.** ABC trilayer aligned to hBN, moire wavelength 9.45 nm
(cell area 77.34 nm^2).  "Filling 4" -- four states, spin x valley, per moire
cell -- is `5.172e12 cm^-2`, reached at `dU = 0.2 eV`.  A single back gate
locks density to displacement field, `n = kappa * dU` with
`kappa = 2.586e17 m^-2 eV^-1`.

That calibration is self-consistent: unscreened single-gate electrostatics
for a 0.2 eV drop across the trilayer gives `4.9e12 cm^-2` at
`eps_graphite = 3`, within 5% of the moire-counting value.

At filling 4 the Landau filling factor is `nu = 14.3` at 15 T, rising to 214
at 1 T.  Since `n ~ dU`, lines of constant `nu` are **straight lines through
the origin** of the (B, dU) plane -- the classic single-gate fan.

`E_F` reaches only 111 meV at `dU = 0.2 eV`, just 26 meV above the conduction
edge: the Mexican hat packs the carriers near the band edge.  That is what
keeps the energy window -- and so the whole map -- affordable.

**Lineshape: use the Gaussian.**  A Lorentzian's tails fall only as 1/E^2, so
the DOS *inside a gap* -- which is exactly what the minima of a Landau fan
are -- picks up contributions from levels arbitrarily far away and never
converges as the retained energy range grows (measured: still drifting
several percent at +-0.9 eV, 7-20% at the ranges one would actually use).
The Gaussian is window-independent to machine precision.  Both are
implemented; `test_lorentzian_tails_do_not_converge` pins the difference.

**Layer projection.**  `solve_auto(..., layers=True)` returns
`Levels.layer_weight`, the `(n_levels, 3)` fraction of each level on layers
1/2/3 (layer 3 = top, at `u3 = +dU/2`), and `dos.dos_layers` projects the DOS
at `E_F` onto them.  This needs eigenvectors, which cost 5-20x more than
eigenvalues alone -- the full 180x150 map goes from ~5 to ~25 minutes.

The valley splitting of the zero multiplet **is** a layer polarisation: at
`dU = 0` the near-zero triplet of K sits 85% on the top layer and that of K'
85% on the bottom, exactly mirrored.  A displacement field pulls them apart
in energy without undoing the locking, though `gamma4`/`delta` break the
mirror by ~8 percentage points at `dU = 0.1 eV`.

The most prominent feature of the map follows from this.  Going up in gate
voltage from neutrality, the states at `E_F` are **bottom**-polarised
(L1 ~ 0.77 at `nu = 1`), then switch abruptly to **top** (L3 ~ 0.89) at
`nu ~ 3`, and depolarise slowly thereafter (L3 ~ 0.50 by `dU = 0.2 eV`, as
`E_F` climbs into the bulk conduction band).

The naive reading -- that electrons added above neutrality go straight into
the upper, top-polarised sextet -- is wrong, and instructively so.  At small
`dU`, `gamma2 = -20 meV` is *larger* than `dU` and reshuffles the multiplet:
at `dU = 9.4 meV` and 10 T the K' triplet sits at (-0.44, +1.88, +5.38) meV
and K at (+7.4, +9.9, +12.8) meV, nothing like `+-dU/2`.  Charge neutrality
therefore lands **inside** the K' triplet, so the first couple of filling
factors go on filling bottom-polarised states before `E_F` clears them.
This is a case where the two-band model, which has no `gamma2`, gets the
layer character of the first few filling factors qualitatively wrong.

## Negative dU: the hole side

Rhombohedral stacking is **centrosymmetric** -- the inversion centre sits in
the middle layer and maps A1 <-> B3, hence K <-> K'.  So

    E(dU, xi) = E(-dU, -xi)       layer weights reverse

exactly (verified to 1e-13 eV and 1e-11 in weight; `test_inversion_symmetry`).
That makes the symmetry a sharp check on the whole Hamiltonian assembly: a
sign slip in `gamma3`/`gamma4`, or a mislabelled layer, breaks it at once.

Negative `dU` is nevertheless **not** a redundant mirror of positive `dU`,
for two reasons.  First, inversion relates the *spectra*, not the *filling*:
with a single gate `n = kappa*dU < 0`, so `dU < 0` is the hole side, which
inversion does not map to the electron side.  Second, once you care about the
top layer specifically, the mirror is exactly what you want to see.

The result is that **the carriers sit on the top layer on both sides**, for
mirror-image reasons: for `dU > 0` the conduction edge is B3-derived (top)
and the added electrons go there; for `dU < 0` the valence edge is B3-derived
and the holes go there.  At `dU = 0` the two valleys balance exactly
(bottom = top = 0.461).

Note the caveat this exposes: in the unscreened linear-drop model the
carriers accumulate on the layer *furthest* from the gate.  Real screening
would push back on that and reduce the effective `dU`; a self-consistent
Hartree treatment is the fix, and is not implemented here.

The solver needed one change for this.  `n(mu)` is monotonic with
`n(0) = 0`, so `sign(mu) == sign(n)` exactly; the adaptive energy range now
opens the valence side when `n < 0`.  The earlier version opened the
conduction side regardless and simply could not find the Fermi level for
`dU < 0`.

## MATLAB export

`abc_ll/matio.py` writes self-describing `.mat` files -- each carries a
`units` struct, a `readme` string, and the full `params` / `gate` /
`broadening` structs that produced it, so a file can be read without the
Python source.  `scipy.io.savemat` preserves shape, so a `(nB, nU)` array
reads as `nB x nU` in MATLAB.

* `scripts/dos_map.py` writes `data/dos_map_<nB>x<nU>.mat`: `B_T`, `dU_eV`,
  `n_cm2`, `nu`, `mu_eV`, `dos_m2_eV`, `dos_layer_m2_eV` (nB x nU x 3,
  ordered bottom/middle/top), `top_fraction`.
* `scripts/export_levels.py` writes the raw levels on three grids
  (`ll_vs_B.mat`, `ll_vs_dU.mat`, `ll_grid.mat`): `energy_eV` as
  `nB x nU x n_max` NaN-padded with an `n_levels` count, plus `layer_weight`
  (`... x 3`, rows summing to 1) and `valley` (+1 for K, -1 for K').

Valleys are merged and sorted by energy in the level exports; use `valley` to
separate them again.

### MATLAB plotting scripts (`matlab/`)

| file | what it does |
|---|---|
| `abc_broadening.m` | **the broadening parameters -- edit this one file** |
| `abc_dos_tools.m` | shared routines: lineshape, DOS, Fermi-level solve |
| `plot_dos_energy_vs_B.m` | DOS with energy on Y, field on X (the spectral fan) |
| `plot_dos_filling_vs_B.m` | DOS at `E_F` with filling on X, field on Y |

The scripts build the DOS **from the raw levels**, applying the broadening
themselves, which is the whole reason the exports carry levels rather than a
pre-broadened DOS: the linewidth stays a live parameter.  `abc_broadening.m`
is the single place it is set, and both scripts call it, so editing that one
file changes every figure.

Base MATLAB only -- no toolboxes.  (In particular `prctile` is deliberately
avoided; it needs Statistics and Machine Learning.)

Two things to know about the colour scaling.  A Gaussian decays to ~1e-50
between levels, so those pixels are numerically nonzero but physically
meaningless; a percentile-based log range would stretch the colour map over
~60 dead decades and the figure comes out uniformly saturated.  Both scripts
therefore floor the log scale a fixed number of `DECADES` below the peak.
And `plot_dos_filling_vs_B.m` shows `dU` as its own panel, because with a
single gate `dU` is not a free axis -- it is set by the filling and the
field, `dU = nu*(e*B/h)/kappa`, and so varies across the plot.

## Performance notes

The eigenvector step of `eig_banded` dominates everything, so two
optimisations matter:

1. **Asymmetric energy ranges.**  In `n(mu) = sum [CDF(mu-E) - CDF(-E)]` a
   level far above `mu` contributes nothing and one far below contributes
   equally at both ends, so only `[0, mu]` plus a few linewidths is needed.
   `e_range=(lo, hi)` exploits that; it both shrinks the basis and cuts the
   number of levels solved for.
2. **`solve_levels_vals`**, an eigenvalue-only path (~20x faster), used by
   `solve_auto` when layer weights are not requested.  It cannot use the
   eigenvector tail filter, so it identifies truncation artefacts as the
   levels that *move* when the basis size changes.

That second one has two traps worth knowing about, both found the hard way:

* **The matching tolerance is load-bearing.**  Genuine levels agree to 0 or
  ~1e-16 eV; artefacts usually move by ~1e-3.  But an artefact that lands
  near a flat valley-polarised zero mode is pinned by the same mechanism as
  the real level, and one was measured drifting only **2e-9 eV**.  At the
  original tolerance of 1e-7 it survived and drew a spurious horizontal
  stripe across the DOS map.  The tolerance is now 1e-11.
* **The detector can be blind.**  Where the artefacts are *exact* zero modes
  -- the two-band models, and `six_band_minimal` -- they are pinned to E = 0
  in every basis size and invisible to energy matching.  `solve_auto` falls
  back to the eigenvector path when it finds no artefacts at all, and the
  fast path is restricted to `FAST_SAFE` (the full six-band model), where
  `test_fast_solver_matches_eigenvector_solver` demonstrates exact agreement
  over 24 cases and a 442-point grid scan found zero mismatches.

## Why the Landau fan lines are straight

A natural worry: the Hamiltonian varies drastically with gate voltage (dU
runs over 400 meV across these maps), so the fan lines ought to warp away
from straight.  They do not, and the reason is worth stating because it is
more general than it first looks.

**Gaps sit at integer filling, whatever the Hamiltonian is doing.**  A
Landau level holds `e*B/h` states per unit area regardless of the
dispersion, so "k levels filled" means `n = 2k * e*B/h` -- a straight line
through the origin of the (n, B) plane.  This is the Streda relation and it
is insensitive to the band structure entirely.  Measured over 405 gaps
spanning B = 2-12 T and dU = 20-200 meV in `six_band_minimal`, every gap
centre lands on an even integer `nu` with deviation **exactly zero**.

So the dU dependence of the Hamiltonian cannot bend a fan line.  What it
controls instead is *which* `nu` are gapped and *how wide* the gaps are:

```
  nu   gaps   min    median    max      (gap width, meV)
   6     42   2.02     3.73   17.52
  10     44   2.05     4.02   19.44
  14     43   2.00     6.95   19.54
```

A tenfold spread in width at fixed `nu`, and lines that come and go: at
B = 8 T, dU = 50 meV there are gaps at nu = 8, 10, 12, 14, 16, 18, but at
dU = 120 meV only nu = 6 and 16 survive.  Level crossings terminate a line;
they never bend it.

**The second reason is an assumption, not physics.**  These maps impose
`n = kappa * dU` exactly, so the gate-voltage-to-density step is linear by
construction.  In a real device it is not -- gate charge splits between
adding carriers and raising E_F (quantum capacitance), which is the
mechanism that would warp a fan in the (V_gate, B) plane.  For this
geometry it is negligible: crossing a 10 meV gap costs ~10 mV out of a
~72 V sweep for a 300 nm oxide, about 0.01%.  A thin top gate, or a much
lower DOS, would make it visible.

**What is actually missing is the moire.**  Streda lines emanate from
*every* B = 0 gap, not only charge neutrality.  A real ABC/hBN sample has
moire band gaps at filling +-4, which launch their own fans from
(filling = +-4, B = 0) -- the satellite fans characteristic of these
systems -- plus Hofstadter structure at higher flux.  This model has only
the dU-induced gap at filling 0, so every line radiates from a single
origin.  That is a direct consequence of treating the sample as weakly
perturbed ABC, and it is the main qualitative feature the model cannot
produce.


## Physics worth knowing before trusting a plot

* The zero-mode multiplet is 12-fold (3 orbital x 2 spin x 2 valley), and a
  displacement field splits it **by valley, not by orbital index**: in the
  minimal model all three zero modes of K sit at exactly `+dU/2` and all three
  of K' at `-dU/2`.  They stay flat in `B`.  The full model keeps this
  structure but moves and disperses the levels.
* Everything here is **single-particle**.  The 12-fold multiplet is exactly the
  degeneracy that interactions are known to lift in rhombohedral graphene; any
  symmetry breaking within it is outside this model.
