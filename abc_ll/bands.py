"""Zero-field band structures for ABC trilayer graphene.

These exist mainly to validate the Landau-level code and to make the
two-band / six-band comparison quantitative: at B = 0 the ladder operators
collapse to the complex number ``v*pi = hbar v (xi k_x + i k_y)`` and both
models become small dense matrices that can be diagonalised on a dense k grid.
"""

from __future__ import annotations

import numpy as np

from .params import ABCParams, layer_potentials


def vpi_scalar(kx, ky, p: ABCParams, valley=1):
    """``v * pi`` in eV for ``pi = xi p_x + i p_y`` at zero field."""
    return p.hbar_v * (valley * np.asarray(kx) + 1j * np.asarray(ky))


def h_two_band_k(kx, ky, du, p: ABCParams, valley=1, minimal=False):
    """Zero-field two-band Hamiltonian; mirrors :func:`landau.h_two_band`."""
    p_ = p.minimal() if minimal else p
    vpi = vpi_scalar(kx, ky, p, valley)
    vpid = np.conj(vpi)
    u1, u2, u3 = layer_potentials(du)
    g1 = p.gamma1
    a2 = np.abs(vpi) ** 2

    x = vpid**3 / g1**2
    h11 = np.full_like(x, u1)
    h22 = np.full_like(x, u3)
    if not minimal:
        x = x + 0.5 * p_.gamma2
        x = x - (p_.gamma3 / p_.gamma0) / g1 * (2.0 * a2)
        h11 = h11 - (u1 - u2) / (2 * g1**2) * a2
        h22 = h22 - (u3 - u2) / (2 * g1**2) * a2

    return np.moveaxis(np.array([[h11, x], [np.conj(x), h22]]), [0, 1], [-2, -1])


def h_six_band_k(kx, ky, du, p: ABCParams, valley=1, minimal=False):
    """Zero-field six-band Hamiltonian; mirrors :func:`landau.h_six_band`."""
    p_ = p.minimal() if minimal else p
    vpi = vpi_scalar(kx, ky, p, valley)
    vpid = np.conj(vpi)
    u1, u2, u3 = layer_potentials(du)
    d = p_.delta
    z = np.zeros_like(vpi)
    o = np.ones_like(vpi)

    v3pi = (p_.gamma3 / p_.gamma0) * vpi
    v4pid = (p_.gamma4 / p_.gamma0) * vpid
    g1 = p_.gamma1 * o
    g2 = 0.5 * p_.gamma2 * o

    rows = [
        [u1 * o, vpid, -v4pid, v3pi, z, g2],
        [None, (u1 + d) * o, g1, -v4pid, z, z],
        [None, None, (u2 + d) * o, vpid, -v4pid, v3pi],
        [None, None, None, (u2 + d) * o, g1, -v4pid],
        [None, None, None, None, (u3 + d) * o, vpid],
        [None, None, None, None, None, u3 * o],
    ]
    for i in range(6):
        for j in range(i):
            rows[i][j] = np.conj(rows[j][i])
    return np.moveaxis(np.array(rows), [0, 1], [-2, -1])


def bands(model: str, kx, ky, du, p: ABCParams, valley=1):
    """Eigenvalues of the requested model on a (broadcast) k grid."""
    builder = {
        "two_band_minimal": lambda: h_two_band_k(kx, ky, du, p, valley, True),
        "two_band": lambda: h_two_band_k(kx, ky, du, p, valley, False),
        "six_band_minimal": lambda: h_six_band_k(kx, ky, du, p, valley, True),
        "six_band": lambda: h_six_band_k(kx, ky, du, p, valley, False),
    }[model]()
    return np.linalg.eigvalsh(builder)


def low_energy_pair(model: str, kx, ky, du, p: ABCParams, valley=1):
    """The two bands straddling charge neutrality (the A1/B3-derived pair)."""
    e = bands(model, kx, ky, du, p, valley)
    nb = e.shape[-1]
    lo = nb // 2 - 1
    return e[..., lo], e[..., lo + 1]
