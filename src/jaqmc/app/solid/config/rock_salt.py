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

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.units import LengthUnit

from .base import LatticeParameters, SolidAtomConfig, SolidConfig

__all__ = ["RockSaltBuilder"]


@configurable_dataclass
class RockSaltBuilder:
    """Builder for a rock-salt crystal.

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

    """

    symbol_a: str = "Li"
    symbol_b: str = "H"
    lattice_constant: float = 4.0
    unit: LengthUnit = LengthUnit.angstrom
    supercell: list[int] | None = None
    s_z: float = 0
    pp: str | dict[str, str] | None = None
    electron_init_width: float = 1.0

    def __call__(self) -> SolidConfig:
        supercell = self.supercell if self.supercell is not None else [1, 1, 1]

        atoms = []
        for symbol, frac_coords in [
            (self.symbol_a, [0.0, 0.0, 0.0]),
            (self.symbol_b, [0.5, 0.5, 0.5]),
        ]:
            atoms.append(SolidAtomConfig(symbol=symbol, frac_coords=frac_coords))

        a = self.lattice_constant / np.sqrt(2)
        return SolidConfig(
            atoms=atoms,
            unit=self.unit,
            lattice=LatticeParameters(a=a, b=a, c=a, alpha=60, beta=60, gamma=60),
            supercell_matrix=[
                [supercell[0], 0, 0],
                [0, supercell[1], 0],
                [0, 0, supercell[2]],
            ],
            s_z=self.s_z,
            pp=self.pp,
            electron_init_width=self.electron_init_width,
        )
