# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rock salt (NaCl) crystal structure geometry builder.

Rock salt has an FCC lattice with two atoms per primitive cell:
- One atom at the origin (0, 0, 0)
- One atom at the body center (L/2, L/2, L/2)

The FCC lattice vectors are::

    a1 = (0, L/2, L/2)
    a2 = (L/2, 0, L/2)
    a3 = (L/2, L/2, 0)
"""

from typing import Any

import numpy as np

from jaqmc.utils.atomic import Atom, get_valence_spin_config
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

from .base import SolidConfig

__all__ = ["rock_salt_config"]


def rock_salt_config(
    symbol_a: str = "Li",
    symbol_b: str = "H",
    lattice_constant: float = 4.0,
    unit: LengthUnit = LengthUnit.angstrom,
    supercell: list[int] | None = None,
    basis: str = "sto-3g",
    ecp: Any = None,
    electron_init_width: float = 1.0,
):
    """Build a rock salt crystal configuration.

    Args:
        symbol_a: Symbol of the atom at the origin.
        symbol_b: Symbol of the atom at the body center.
        lattice_constant: Lattice constant.
        unit: Unit of the lattice constant ('angstrom' or 'bohr').
        supercell: Supercell dimensions [nx, ny, nz]. Defaults to [1, 1, 1].
        basis: Basis set name for HF pretrain.
        ecp: Effective core potential specification. Can be ``None``
            (all-electron), a string (e.g., ``"ccecp"``), or a
            per-element mapping (e.g., ``{"Li": "ccecp"}``).
        electron_init_width: Width for electron position initialization.

    Returns:
        A SolidConfig instance for the rock salt structure.
    """
    if supercell is None:
        supercell = [1, 1, 1]

    L = lattice_constant
    if unit == LengthUnit.angstrom:
        L *= ONE_ANGSTROM_IN_BOHR

    # FCC primitive lattice vectors
    lattice_vectors = (np.ones((3, 3)) - np.eye(3)) * L / 2

    # Atoms in primitive cell
    atoms = []
    n_electrons = 0
    for symbol, coords in [
        (symbol_a, [0.0, 0.0, 0.0]),
        (symbol_b, [L / 2, L / 2, L / 2]),
    ]:
        if ecp is not None:
            valence = sum(get_valence_spin_config(symbol, ecp))
            atom = Atom(symbol=symbol, coords=coords, charge=valence)
            n_electrons += valence
        else:
            atom = Atom(symbol=symbol, coords=coords)
            n_electrons += atom.atomic_number
        atoms.append(atom)

    # Compute electron spins with net spin = 0 (closed shell)
    n_up = n_electrons // 2
    n_down = n_electrons - n_up

    return SolidConfig(
        atoms=atoms,
        lattice_vectors=lattice_vectors.tolist(),
        supercell_matrix=[
            [supercell[0], 0, 0],
            [0, supercell[1], 0],
            [0, 0, supercell[2]],
        ],
        electron_spins=(n_up, n_down),
        basis=basis,
        ecp=ecp,
        electron_init_width=electron_init_width,
    )
