# Open issue: the Berry / orbital-moment sign on the conduction band

Status at 2026-10-18. The semiclassical fan matches the exact levels well on
the **valence** side of every model and poorly on the **conduction** side.
This note records what has been ruled in and out, so the next session does not
repeat it.

## Measured

Negating both `Oz` and `Lz` (equivalently `term_factors = (-1,-1,0)`), against
the engine defaults, dU = 100 meV, B = 9 T, median |dE| to the nearest exact
level over the upper half of the levels, in meV:

| model | valley | band | as-is | negated |
|---|---|---|---|---|
| six_band_minimal | K  | cond | 4.834 | **1.329** |
| six_band_minimal | K  | val  | **0.965** | 5.396 |
| six_band_minimal | Kp | cond | 6.165 | **1.742** |
| six_band_minimal | Kp | val  | **0.839** | 5.308 |
| six_band         | K  | cond | 4.665 | **3.588** |
| six_band         | K  | val  | **0.715** | 5.617 |
| six_band         | Kp | cond | 6.737 | **2.434** |
| six_band         | Kp | val  | **1.305** | 3.832 |

**Valence wants as-is, conduction wants negated, in both valleys.**  The split
is by band, not by valley, and it is consistent across valleys and models.

## Ruled out

* **A valley-label swap.**  An earlier single-point comparison suggested the
  semiclassical valley-K fan reproduced exact K' (1.02 vs 3.20 meV).  The table
  above kills that: the pattern is identical in the two valleys, so it is not a
  valley mix-up.  That earlier reading was one point, conduction only, and it
  does not survive contact with the other seven rows.
* **A bug in Oz or Lz.**  `Lz = -(dE/2)*Oz` holds to 0.00e+00 for the two-band
  model (the exact two-band identity), and the engine's thermal `dL/dE` kernel
  reproduces a finite difference of `L(mu)` to 1e-4.  Both evaluations are right.
* **A wrong orbit area.**  The bare Onsager level spacing from `dS/dE` matches
  the exact spacing to <1% for both six-band models.
* **`Bmultiplier`.**  Pinned to 1 analytically; see `tests/test_semiclassical.py`.
* **A clean factor on the orbital-moment term.**  `morb = 0.5*bc_sign`, derived
  from `Delta_bracket = -(phi0/(2pi)^2)(dS/dmu)m`, scores 0.434 on the two-band
  conduction band against 0.043 for `morb = +1`.  The derivation is wrong
  somewhere; `morb = +1` is right for the model that can be checked analytically.

## Where it stands

`export_semiclassical.bc_sign()` currently flips **only** the Berry term per
orbit.  The table says both terms want flipping together on the conduction
side.  That is consistent with the traversal argument -- `hbar kdot = -e v x B`
reverses the sense of circulation when `grad E` points inward -- which should
affect every first-order-in-B term alike, not just the Berry phase.

Next step: set `term_factors = (s, s, 0)` with `s = bc_sign(...)`, re-run the
export and `compare_semiclassical.py`, and check whether conduction comes down
to the valence level (~0.05 of a spacing) or stalls around 0.1.  If it stalls,
the residual is likely genuine higher-order semiclassics: the zero-field bands
of `six_band_minimal` are exactly particle-hole symmetric (max|Ec+Ev| = 0.000
meV) while its exact valley-resolved LL spectrum is **not**, so no choice of
prefactor built on those bands can reproduce the asymmetry.

Until this is settled, trust `two_band_minimal` (both bands) and the valence
side of the six-band models; do not trust six-band conduction.
