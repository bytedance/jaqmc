# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""1D two-atom chain geometry builder.

A one-dimensional chain with two atoms per primitive cell along the x-axis.
The primitive cell has atoms at (0, 0, 0) and (bond_length, 0, 0), with
lattice constant 2 * bond_length. The lattice is made effectively 1D by
using very large lattice constants in y and z directions.
"""

from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

from .base import LatticeDirect, SolidAtomConfig, SolidConfig

__all__ = ["two_atom_chain"]


def two_atom_chain(
    symbol: str = "H",
    bond_length: float = 1.8,
    unit: LengthUnit = LengthUnit.bohr,
    supercell: int = 1,
    vacuum_separation: float = 100.0,
    s_z: float = 0,
    pp: str | dict[str, str] | None = None,
    electron_init_width: float = 1.0,
):
    """Build a 1D two-atom chain configuration.

    Args:
        symbol: Atomic symbol.
        bond_length: Distance between atoms along the chain.
        unit: Unit of the bond length ('angstrom' or 'bohr').
        supercell: Supercell expansion factor along the chain direction.
        vacuum_separation: Lattice constant in y and z directions (in Bohr)
            to isolate the 1D chain.
        s_z: Total spin along the z direction of the explicit primitive-cell
            electrons.
        pp: Pseudopotential specification. Can be ``None`` (all-electron),
            a string (e.g., ``"ccecp"``), or a per-element mapping
            (e.g., ``{"Li": "ccecp"}``).
        electron_init_width: Width for electron position initialization.

    Returns:
        A SolidConfig instance for the two-atom chain.
    """
    if unit == LengthUnit.angstrom:
        bond_length *= ONE_ANGSTROM_IN_BOHR

    frac_coords_list = [
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]

    atoms = [
        SolidAtomConfig(symbol=symbol, frac_coords=frac_coords)
        for frac_coords in frac_coords_list
    ]
    return SolidConfig(
        atom_configs=atoms,
        lattice=LatticeDirect(
            a=(2 * bond_length, 0.0, 0.0),
            b=(0.0, vacuum_separation, 0.0),
            c=(0.0, 0.0, vacuum_separation),
        ),
        supercell_matrix=[[supercell, 0, 0], [0, 1, 0], [0, 0, 1]],
        s_z=s_z,
        pp=pp,
        electron_init_width=electron_init_width,
    )
