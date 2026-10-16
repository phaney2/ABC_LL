"""Stamp the charge-neutrality energy into the exported .mat files.

Charge neutrality is not at E = 0 -- `delta` and `gamma4` break the chiral
symmetry of the ABC chain and push the low-energy structure to about
+6.3 meV at dU = 0 (see `dos.cnp_energy`).  The MATLAB side needs that
reference to convert between chemical potential and filling.

The Landau levels themselves do not depend on it, so this only adds a field
rather than recomputing anything.

Run:  python3 scripts/add_cnp_to_mat.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc_ll import dos  # noqa: E402
from abc_ll.params import DEFAULT as P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data")

NOTE = (" Charge neutrality is NOT at E = 0: delta and gamma4 shift it to "
        "about +6.3 meV at dU = 0. cnp_eV holds it, with the same shape as "
        "the dU grid. Measure filling from cnp_eV, not from zero.")


def stamp(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  {name}: not found, skipping")
        return
    m = loadmat(path, squeeze_me=False, struct_as_record=True)
    m = {k: v for k, v in m.items() if not k.startswith("__")}

    du = np.atleast_1d(np.squeeze(m["dU_eV"]))
    flat = np.array([dos.cnp_energy(float(x), P) for x in du.ravel()])
    cnp = flat.reshape(du.shape)
    m["cnp_eV"] = cnp
    m["cnp_meV"] = cnp * 1e3
    if "readme" in m:
        txt = str(np.squeeze(m["readme"]))
        if "cnp_eV" not in txt:
            m["readme"] = txt + NOTE
    savemat(path, m, do_compression=True)
    print(f"  {name}: cnp_eV {cnp.shape}, "
          f"{cnp.min()*1e3:+.3f} to {cnp.max()*1e3:+.3f} meV")


if __name__ == "__main__":
    for f in ("dos_map_120x300.mat", "ll_vs_B.mat", "ll_vs_dU.mat",
              "ll_grid.mat", "ll_vs_nu.mat"):
        stamp(f)
