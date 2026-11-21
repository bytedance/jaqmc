# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""1D two-atom chain geometry builder.

A one-dimensional chain with two atoms per primitive cell along the x-axis.
The primitive cell has atoms at (0, 0, 0) and (bond_length, 0, 0), with
lattice constant 2 * bond_length. The lattice is made effectively 1D by
using very large lattice constants in y and z directions.
"""

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

from .base import LatticeVectors, SolidAtomConfig, SolidConfig

__all__ = ["TwoAtomChainBuilder"]


@configurable_dataclass
class TwoAtomChainBuilder:
    """Configuration for a one-dimensional two-atom chain.

    Args:
        symbol: Atomic symbol for both atoms in the primitive cell.
        bond_length: Distance between the two atoms along the chain.
        unit: Length unit for ``bond_length``.
        supercell: Repetition factor along the chain direction.
        vacuum_separation: Transverse cell size in Bohr along y and z.
        s_z: Total spin along the z direction of the explicit primitive-cell
            electrons.
        pp: Pseudopotential specification. Can be ``None`` (all-electron),
            a string (e.g., ``"ccecp"``), or a per-element mapping
            (e.g., ``{"H": "ccecp"}``).
        electron_init_width: Width of Gaussian for electron initialization.
    """

    symbol: str = "H"
    bond_length: float = 1.8
    unit: LengthUnit = LengthUnit.bohr
    supercell: int = 1
    vacuum_separation: float = 100.0
    s_z: float = 0
    pp: str | dict[str, str] | None = None
    electron_init_width: float = 1.0

    def __call__(self) -> SolidConfig:
        bond_length = self.bond_length
        if self.unit == LengthUnit.angstrom:
            bond_length *= ONE_ANGSTROM_IN_BOHR

        atoms = [
            SolidAtomConfig(symbol=self.symbol, frac_coords=frac_coords)
            for frac_coords in ([0.0, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        return SolidConfig(
            atoms=atoms,
            lattice=LatticeVectors(
                a=(2 * bond_length, 0.0, 0.0),
                b=(0.0, self.vacuum_separation, 0.0),
                c=(0.0, 0.0, self.vacuum_separation),
            ),
            supercell_matrix=[[self.supercell, 0, 0], [0, 1, 0], [0, 0, 1]],
            s_z=self.s_z,
            pp=self.pp,
            electron_init_width=self.electron_init_width,
        )
