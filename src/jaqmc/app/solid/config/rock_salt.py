# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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

import numpy as np

from jaqmc.utils.units import LengthUnit

from .base import LatticeParams, SolidAtomConfig, SolidConfig

__all__ = ["rock_salt_config"]


def rock_salt_config(
    symbol_a: str = "Li",
    symbol_b: str = "H",
    lattice_constant: float = 4.0,
    unit: LengthUnit = LengthUnit.angstrom,
    supercell: list[int] | None = None,
    s_z: float = 0,
    pp: str | dict[str, str] | None = None,
    electron_init_width: float = 1.0,
):
    """Build a rock salt crystal configuration.

    Args:
        symbol_a: Symbol of the atom at the origin.
        symbol_b: Symbol of the atom at the body center.
        lattice_constant: Lattice constant.
        unit: Unit of the lattice constant ('angstrom' or 'bohr').
        supercell: Supercell dimensions [nx, ny, nz]. Defaults to [1, 1, 1].
        s_z: Total spin along the z direction of the explicit primitive-cell
            electrons.
        pp: Pseudopotential specification. Can be ``None`` (all-electron),
            a string (e.g., ``"ccecp"``), or a per-element mapping
            (e.g., ``{"Li": "ccecp"}``).
        electron_init_width: Width for electron position initialization.

    Returns:
        A SolidConfig instance for the rock salt structure.
    """
    if supercell is None:
        supercell = [1, 1, 1]

    atoms = []
    for symbol, frac_coords in [
        (symbol_a, [0.0, 0.0, 0.0]),
        (symbol_b, [0.5, 0.5, 0.5]),
    ]:
        atoms.append(SolidAtomConfig(symbol=symbol, frac_coords=frac_coords))

    a = lattice_constant / np.sqrt(2)
    return SolidConfig(
        atom_configs=atoms,
        unit=unit,
        lattice=LatticeParams(a=a, b=a, c=a, alpha=60, beta=60, gamma=60),
        supercell_matrix=[
            [supercell[0], 0, 0],
            [0, supercell[1], 0],
            [0, 0, supercell[2]],
        ],
        s_z=s_z,
        pp=pp,
        electron_init_width=electron_init_width,
    )
