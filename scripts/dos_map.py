"""Density of states at the Fermi level over the (B, dU) plane.

Single back gate: the gate voltage fixes both the displacement field and the
carrier density, so n and dU are locked (n = kappa * dU) and the map is a
Landau fan.  At fixed gate voltage the density does *not* move with B; the
chemical potential does.

Because n is proportional to dU, the filling factor is

    nu = n h / (e B) = kappa * dU / (e B / h)

so lines of constant nu are straight lines dU ~ B through the origin -- the
classic single-gate fan.

Produces:
  dos_map.png        DOS(E_F) and its top-layer projection over (B, dU)
  dos_cuts.png       line cuts through the map

Run:  python3 scripts/dos_map.py        (env: NB, NU set the grid size)
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import dos, matio, plotstyle  # noqa: E402
from abc_ll.params import DEFAULT as P, NPHI_1T  # noqa: E402

#: The parameter set must match the model: the minimal models have
#: delta = gamma4 = 0, and pairing them with the full set would give the
#: wrong charge-neutrality reference.
if os.environ.get("MODEL", "six_band").endswith("minimal"):
    P = P.minimal()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "data")
MEV = 1e3

MODEL = os.environ.get("MODEL", "six_band")
GATE = dos.moire_gate()                      # 4 states / 9.45 nm cell at 0.2 eV
#: Gaussian, not Lorentzian: a Lorentzian's 1/E^2 tails make the in-gap DOS --
#: which is exactly what the fan minima are -- depend on how many distant
#: levels are retained.  See `dos.Broadening`.
BROAD = dos.Broadening(gamma0=1.0e-3, scaling="sqrtB", lineshape="gaussian")

#: DOS unit for plotting: 1e12 states per cm^2 per eV, from m^-2 eV^-1
DOS_UNIT = 1e12 * 1e4


def _point(args):
    B, du = args
    try:
        mu, d, dl, _ = dos.fermi_level_and_dos(MODEL, B, du, P, GATE, BROAD,
                                               layers=True)
    except RuntimeError:
        return (np.nan,) * 5
    return (mu, d, dl[0], dl[1], dl[2])


def compute(B, du, workers=None, cache=None):
    """mu, DOS(E_F) and the layer-resolved DOS on the (B, du) grid."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z["B"].shape == B.shape and z["du"].shape == du.shape:
            print("  (loaded from cache)")
            return z["mu"], z["dos"], z["dos_layer"]
    pts = [(b, d) for b in B for d in du]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_point, pts, chunksize=8))
    print(f"  {len(pts)} points in {time.perf_counter()-t0:.0f} s")
    res = np.array(res).reshape(len(B), len(du), 5)
    mu, d, dl = res[..., 0], res[..., 1], res[..., 2:]
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, B=B, du=du, mu=mu, dos=d, dos_layer=dl)
    return mu, d, dl


def nu_of(B, du):
    return GATE.density(du) / (NPHI_1T * B)


def _nu_lines(ax, B, du, color):
    """Constant-filling lines, which for a single gate radiate from the origin.

    dU < 0 is the hole side (n = kappa*dU < 0), so the fan is two-sided.
    """
    lo, hi = du[0] * MEV, du[-1] * MEV
    for nu in (4, 8, 12, 16, 24, 32, 48, 64):
        for sign in (+1, -1):
            if sign > 0 and hi <= 0:
                continue
            if sign < 0 and lo >= 0:
                continue
            y = sign * nu * NPHI_1T * B / GATE.kappa * MEV
            if sign > 0 and y[0] > hi:
                continue
            if sign < 0 and y[0] < lo:
                continue
            ax.plot(B, y, color=color, lw=0.7, alpha=0.5, zorder=3)
            # label where the line is still clear of the edge, so steep lines
            # do not push their label outside the axes
            inside = (np.where(y <= 0.9 * hi)[0] if sign > 0
                      else np.where(y >= 0.9 * lo)[0])
            if inside.size:
                i = inside[-1]
                ax.annotate(rf"$\nu$={sign*nu:+d}", xy=(B[i], y[i]),
                            xytext=(-2, 3 if sign > 0 else -9),
                            textcoords="offset points", fontsize=6.5,
                            ha="right", color=color, zorder=4, alpha=0.9)
    if lo < 0 < hi:
        ax.axhline(0, color=color, lw=0.7, alpha=0.6, zorder=3)
    ax.set_ylim(lo, hi)


def _colorbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(label, color=plotstyle.INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, color=plotstyle.MUTED)
    cb.outline.set_visible(False)
    return cb


# ------------------------------------------------------------------ map
def fig_map(B, du, d, mu, dl):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True,
                             sharey=True)
    extent = [B[0], B[-1], du[0] * MEV, du[-1] * MEV]

    ax = axes[0]
    z = d.T / DOS_UNIT
    im = ax.imshow(z, origin="lower", aspect="auto", extent=extent,
                   interpolation="nearest",
                   cmap=plotstyle.sequential_cmap(hue="blue"),
                   norm=LogNorm(vmin=np.nanpercentile(z, 0.5),
                                vmax=np.nanpercentile(z, 99.8)))
    _colorbar(fig, ax, im,
              r"DOS at $E_F$   ($10^{12}$ cm$^{-2}$ eV$^{-1}$)")
    _nu_lines(ax, B, du, plotstyle.SURFACE)
    ax.set_title("DOS at the Fermi level", loc="left")
    ax.set_xlabel("magnetic field  B  (T)")
    ax.set_ylabel(r"interlayer potential  $\Delta U$  (meV)")
    ax.grid(False)

    ax = axes[1]
    top = (dl[..., 2] / dl.sum(axis=-1)).T
    im = ax.imshow(top, origin="lower", aspect="auto", extent=extent,
                   interpolation="nearest", vmin=0.0, vmax=1.0,
                   cmap=plotstyle.sequential_cmap("abc_orange", hue="orange"))
    _colorbar(fig, ax, im, r"fraction of DOS($E_F$) on the top layer")
    _nu_lines(ax, B, du, plotstyle.INK)
    ax.set_title(r"Top-layer projection of the states at $E_F$", loc="left")
    ax.set_xlabel("magnetic field  B  (T)")
    ax.grid(False)

    nu_max = nu_of(B[0], du[-1])
    fig.suptitle(
        "ABC trilayer, single back gate: n and "
        rf"$\Delta U$ locked ({GATE.n_max/1e16:.2f}$\times10^{{12}}$ "
        r"cm$^{-2}$ at $\Delta U$ = 200 meV, i.e. 4 states per 9.45 nm moire "
        rf"cell).  Constant-$\nu$ lines radiate from the origin; $\nu$ "
        rf"reaches {nu_max:.0f} at top left.",
        x=0.005, ha="left", fontsize=9.5, color=plotstyle.INK,
        weight="semibold")
    fig.savefig(os.path.join(FIGDIR, "dos_map.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- cuts
def fig_cuts(B, du, d, mu, dl):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)

    ax = axes[0]
    for i, du0 in enumerate((-0.12, 0.05, 0.12)):
        j = int(np.argmin(np.abs(du - du0)))
        ax.plot(B, d[:, j] / DOS_UNIT, color=plotstyle.SERIES[i], lw=1.1,
                label=rf"$\Delta U$ = {du[j]*MEV:.0f} meV")
    ax.set_yscale("log")
    ax.set_xlabel("magnetic field  B  (T)")
    ax.set_ylabel(r"DOS at $E_F$  ($10^{12}$ cm$^{-2}$ eV$^{-1}$)")
    ax.set_title("vs field, at fixed gate voltage", loc="left")
    ax.legend(handlelength=1.4)

    ax = axes[1]
    for i, B0 in enumerate((5.0, 10.0, 15.0)):
        k = int(np.argmin(np.abs(B - B0)))
        ax.plot(du * MEV, d[k, :] / DOS_UNIT, color=plotstyle.SERIES[i],
                lw=1.1, label=f"B = {B[k]:.0f} T")
    ax.set_yscale("log")
    ax.set_xlabel(r"interlayer potential  $\Delta U$  (meV)")
    ax.set_title("vs gate voltage (= density = displacement field)", loc="left")
    ax.legend(handlelength=1.4)

    ax = axes[2]
    k = int(np.argmin(np.abs(B - 15.0)))
    frac = dl[k] / dl[k].sum(axis=-1, keepdims=True)
    for i, name in enumerate(("layer 1 (bottom)", "layer 2 (middle)",
                              "layer 3 (top)")):
        ax.plot(du * MEV, frac[:, i], color=plotstyle.SERIES[i], lw=1.1,
                label=name)
    ax.axhline(1 / 3, color=plotstyle.AXIS, lw=0.7, ls=(0, (4, 3)))
    ax.axvline(0, color=plotstyle.AXIS, lw=0.7)
    ax.annotate("unpolarised", xy=(du[-1] * MEV, 1 / 3), xytext=(-3, 4),
                textcoords="offset points", ha="right", fontsize=7,
                color=plotstyle.MUTED)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"interlayer potential  $\Delta U$  (meV)")
    ax.set_ylabel(r"fraction of DOS($E_F$)")
    ax.set_title(f"layer projection at B = {B[k]:.0f} T", loc="left")
    ax.legend(handlelength=1.4, loc="upper right")

    fig.suptitle(
        "Cuts: DOS minima mark gaps between Landau levels.  The top layer "
        r"carries most of the weight on $\it{both}$ sides of neutrality -- "
        r"electrons enter the B3-derived conduction edge, holes the "
        r"B3-derived valence edge -- except in a narrow window just above "
        r"$\Delta U = 0$ where $\gamma_2$ puts $E_F$ inside the "
        r"bottom-polarised K' triplet",
        x=0.005, ha="left", fontsize=10, color=plotstyle.INK,
        weight="semibold")
    fig.savefig(os.path.join(FIGDIR, "dos_cuts.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    plotstyle.apply()
    nB = int(os.environ.get("NB", 180))
    nU = int(os.environ.get("NU", 300))
    B = np.linspace(float(os.environ.get("BMIN", 1.0)),
                    float(os.environ.get("BMAX", 15.0)), nB)
    du = np.linspace(-0.20, 0.20, nU)
    print(f"grid {nB} x {nU}   model={MODEL}   broadening={BROAD.lineshape}")
    mu, d, dl = compute(B, du, cache=os.path.join(CACHE, f"dos_{MODEL}_{nB}x{nU}.npz"))
    bad = int(np.sum(~np.isfinite(d)))
    print(f"  {bad} failed points" if bad else "  all points converged")
    fig_map(B, du, d, mu, dl)
    fig_cuts(B, du, d, mu, dl)
    print("figures written")
    mat = os.path.join(CACHE, f"dos_map_{MODEL}_{nB}x{nU}.mat")
    matio.save_dos_map(mat, B, du, mu, d, dl, P, GATE, BROAD, MODEL)
    print(f"wrote {mat}")
