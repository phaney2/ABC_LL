"""Continuum-model parameters for ABC (rhombohedral) trilayer graphene.

Conventions used throughout this package
----------------------------------------
* **Energies are in eV.**  Every term in the Hamiltonian is an energy, so we
  never carry a velocity in m/s.  The only place the Fermi velocity appears is
  the combination ``hbar*v = (sqrt(3)/2) * a * gamma0``  [eV m], which divided
  by the magnetic length gives the Landau-level energy scale.
* **Lengths are in metres**, densities in m^-2 (helpers convert to cm^-2).
* **Sublattice basis** is ``(A1, B1, A2, B2, A3, B3)``.
* ABC stacking: layer 2 is displaced from layer 1 by one bond vector, layer 3
  by two.  Hence the *dimer* pairs are (B1, A2) and (B2, A3), the low-energy
  sites are A1 and B3, and A1 sits directly below B3 -- that vertical pair is
  the one coupled by ``gamma2``.  (There is no ``gamma5`` in ABC: unlike ABA,
  no A-A pair is vertically aligned across layers 1 and 3.)
* **Valley** ``xi = +1`` is K, ``xi = -1`` is K'.  We define the momentum
  operator ``pi = xi * pi_x + i * pi_y`` with ``pi_vec = p_vec + e A_vec``.
* **Layer potentials** for a linear potential drop are
  ``(u1, u2, u3) = (-du/2, 0, +du/2)`` where ``du = u3 - u1`` is the total
  interlayer potential difference (the quantity the user caps at 0.2 eV).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------- constants
HBAR_EVS = 6.582119569e-16      # hbar  [eV s]
HBAR_SI = 1.054571817e-34       # hbar  [J s]
E_CHARGE = 1.602176634e-19      # e     [C]
H_PLANCK_SI = 6.62607015e-34    # h     [J s]
PHI0 = H_PLANCK_SI / E_CHARGE   # flux quantum h/e [Wb]

#: magnetic length at B = 1 T, in metres:  l_B = LB_1T / sqrt(B[T])
LB_1T = np.sqrt(HBAR_SI / E_CHARGE)     # 25.6556 nm

#: Landau-level degeneracy per unit area, per spin per valley, at B = 1 T
#: (n_phi = e B / h), in m^-2 T^-1.
NPHI_1T = E_CHARGE / H_PLANCK_SI        # 2.41799e14 m^-2 T^-1

M2_PER_CM2 = 1e4                        # multiply m^-2 by this -> ... no:
                                        # n[cm^-2] = n[m^-2] / 1e4


def to_cm2(n_m2):
    """Convert an areal density from m^-2 to cm^-2."""
    return np.asarray(n_m2) / 1e4


def to_m2(n_cm2):
    """Convert an areal density from cm^-2 to m^-2."""
    return np.asarray(n_cm2) * 1e4


@dataclass(frozen=True)
class ABCParams:
    """Slonczewski-Weiss-McClure style hopping parameters (eV) for ABC TLG.

    Defaults are the Koshino-McCann trilayer set, which is the one most
    commonly used for rhombohedral trilayer continuum calculations.

    Attributes
    ----------
    gamma0 : nearest-neighbour intralayer hopping.
    gamma1 : vertical dimer coupling, B1-A2 and B2-A3.
    gamma2 : coupling between the vertically aligned A1 and B3 sites.  Enters
             the Hamiltonian as ``gamma2 / 2`` (standard SWMc convention).
             Negative by convention; sets the ~10 meV scale on which the cubic
             band touching splits into separate Dirac points.
    gamma3 : skew interlayer coupling between the two *non-dimer* sites of a
             neighbouring layer pair (A1-B2 and A2-B3).  Trigonal warping.
    gamma4 : skew interlayer coupling A1-A2, B1-B2, A2-A3, B2-B3.  Breaks
             electron-hole symmetry.
    delta  : on-site energy of the dimer sites (B1, A2, B2, A3) relative to the
             non-dimer sites (A1, B3).
    a      : graphene lattice constant [m].
    d0     : interlayer spacing [m] (only used for electrostatics helpers).
    """

    gamma0: float = 3.16
    gamma1: float = 0.39
    gamma2: float = -0.020
    gamma3: float = 0.315
    gamma4: float = 0.044
    delta: float = 0.0366
    a: float = 0.246e-9
    d0: float = 0.335e-9

    # ---------------------------------------------------------- derived
    @property
    def hbar_v(self) -> float:
        """hbar * v_F  [eV m].  v = sqrt(3) a gamma0 / (2 hbar)."""
        return 0.5 * np.sqrt(3.0) * self.a * self.gamma0

    @property
    def v_ms(self) -> float:
        """Fermi velocity in m/s (diagnostic only)."""
        return self.hbar_v / HBAR_EVS

    @property
    def cell_area(self) -> float:
        """Graphene unit-cell area [m^2]  =  sqrt(3)/2 * a^2."""
        return 0.5 * np.sqrt(3.0) * self.a**2

    def omega(self, B):
        """Dirac Landau-level energy scale  sqrt(2) hbar v / l_B  [eV].

        This is the n=1 monolayer Landau level energy; ~36.9 meV * sqrt(B[T])
        for the default parameters.
        """
        return np.sqrt(2.0) * self.hbar_v * np.sqrt(np.asarray(B, float)) / LB_1T

    def cubic_coeff(self) -> float:
        """Coefficient of (hbar v k)^3 in the two-band model: 1 / gamma1^2 [1/eV^2]."""
        return 1.0 / self.gamma1**2

    def with_(self, **kw) -> "ABCParams":
        """Return a copy with some fields replaced."""
        return replace(self, **kw)

    def minimal(self) -> "ABCParams":
        """Copy with all remote hoppings and the dimer on-site shift switched off."""
        return replace(self, gamma2=0.0, gamma3=0.0, gamma4=0.0, delta=0.0)


#: Alternative parameter set: graphite SWMc values used by Zhang, Jung, Fiete,
#: Niu & MacDonald, PRB 82, 035409 (2010) for ABC trilayer.
ZHANG2010 = ABCParams(
    gamma0=3.16,
    gamma1=0.502,
    gamma2=-0.0171,
    gamma3=-0.377,
    gamma4=-0.099,
    delta=-0.0014,
)

DEFAULT = ABCParams()


def layer_potentials(du: float):
    """Linear potential drop across the three layers.

    Parameters
    ----------
    du : total interlayer potential difference ``u3 - u1`` in eV.

    Returns
    -------
    (u1, u2, u3) with the middle layer at zero, so the array is traceless.
    """
    return -0.5 * du, 0.0, 0.5 * du


def nphi(B):
    """Landau-level degeneracy per unit area per spin per valley [m^-2]."""
    return NPHI_1T * np.asarray(B, float)


def filling_factor(n_m2, B):
    """Landau filling factor nu = n h / (e B), counting *all* spin/valley flavours."""
    return np.asarray(n_m2, float) / nphi(B)
