# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""1D two-atom chain geometry builder.

A one-dimensional chain with two atoms per primitive cell along the x-axis.
The primitive cell has atoms at (0, 0, 0) and (bond_length, 0, 0), with
lattice constant 2 * bond_length. The lattice is made effectively 1D by
using very large lattice constants in y and z directions.
"""

from typing import Any

from jaqmc.utils.atomic import Atom, get_valence_spin_config
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

from .base import SolidConfig

__all__ = ["two_atom_chain"]


def two_atom_chain(
    symbol: str = "H",
    bond_length: float = 1.8,
    unit: LengthUnit = LengthUnit.bohr,
    supercell: int = 1,
    vacuum_separation: float = 100.0,
    spin: int = 0,
    basis: str = "sto-3g",
    ecp: Any = None,
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
        spin: Total spin polarization (n_up - n_down) for the primitive cell.
        basis: Basis set name for HF pretrain.
        ecp: Effective core potential specification. Can be ``None``
            (all-electron), a string (e.g., ``"ccecp"``), or a
            per-element mapping (e.g., ``{"Li": "ccecp"}``).
        electron_init_width: Width for electron position initialization.

    Returns:
        A SolidConfig instance for the two-atom chain.

    Raises:
        ValueError: Electrons per unit cell and given spin have different parity.
    """
    if unit == LengthUnit.angstrom:
        bond_length *= ONE_ANGSTROM_IN_BOHR

    # 1D lattice vectors (large separation in y and z)
    lattice_vectors = [
        [2 * bond_length, 0.0, 0.0],
        [0.0, vacuum_separation, 0.0],
        [0.0, 0.0, vacuum_separation],
    ]

    yz_center = vacuum_separation / 2
    coords_list = [
        [0.0, yz_center, yz_center],
        [bond_length, yz_center, yz_center],
    ]

    atoms = []
    electrons = 0
    for coords in coords_list:
        if ecp is not None:
            valence = sum(get_valence_spin_config(symbol, ecp))
            atom = Atom(symbol=symbol, coords=coords, charge=valence)
            electrons += valence
        else:
            atom = Atom(symbol=symbol, coords=coords)
            electrons += atom.atomic_number
        atoms.append(atom)

    if (electrons + spin) % 2 != 0:
        raise ValueError(f"Impossible to have spin {spin} for {electrons} electrons.")
    # Compute electron spins per primitive cell
    n_up = (electrons + spin) // 2
    n_down = (electrons - spin) // 2

    return SolidConfig(
        atoms=atoms,
        lattice_vectors=lattice_vectors,
        supercell_matrix=[[supercell, 0, 0], [0, 1, 0], [0, 0, 1]],
        electron_spins=(n_up, n_down),
        basis=basis,
        ecp=ecp,
        electron_init_width=electron_init_width,
    )
