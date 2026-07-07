# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal

import serde

from jaqmc.utils.atomic import Atom, AtomicSystemConfig, AtomInitialization
from jaqmc.utils.atomic.pretrain import PretrainReferenceConfig
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

__all__ = ["MoleculeConfig", "MoleculePretrainReferenceConfig"]


@configurable_dataclass(kw_only=False)
class AtomConfig:
    """Input configuration for one atom before system-level resolution.

    Args:
        symbol: Chemical element symbol (e.g. ``"H"``, ``"Fe"``).
        coords: 3D Cartesian coordinates in the unit specified by the containing
            :class:`~jaqmc.utils.atomic.atomic_system.AtomicSystemConfig`.
        charge: Optional explicit effective charge seen by the simulated electrons.
            When omitted, it is derived from the element and the configured
            pseudopotential treatment.
        initialization: Optional per-atom hints used only for electron initialization.
    """

    symbol: str
    coords: list[float]
    charge: int | None = None
    initialization: AtomInitialization = field(default_factory=AtomInitialization)

    def __post_init__(self):
        if len(self.coords) != 3:
            raise ValueError(
                f"Expected three-dimensional atom. Got {len(self.coords)} in {self}."
            )


@configurable_dataclass
class MoleculeConfig(AtomicSystemConfig):
    """Configuration for arbitrary molecules.

    Args:
        unit: Length unit used by the geometry inputs.
        atom_configs: Atoms in the molecule, given as a list of
            :class:`AtomConfig` entries.
        electron_init_width: Width of the Gaussian distribution for
            initializing electron positions.
    """

    unit: LengthUnit = LengthUnit.bohr
    atom_configs: list[AtomConfig] = serde.field(
        rename="atoms",
        default_factory=lambda: [AtomConfig(symbol="He", coords=[0, 0, 0])],
    )
    electron_init_width: float = 1.0

    @property
    def atoms(self) -> list[Atom]:
        return [
            Atom(
                symbol=atom.symbol,
                coords=[c * ONE_ANGSTROM_IN_BOHR for c in atom.coords]
                if self.unit == LengthUnit.angstrom
                else atom.coords,
                charge=atom.charge,
                pp=self.pp.get(atom.symbol) if isinstance(self.pp, dict) else self.pp,
            )
            for atom in self.atom_configs
        ]

    @property
    def per_atom_init(self) -> list[AtomInitialization]:
        return [atom.initialization for atom in self.atom_configs]


@configurable_dataclass
class MoleculePretrainReferenceConfig(PretrainReferenceConfig):
    method: Literal["RHF", "UHF"] = "UHF"
    "Variants of Hartree-Fock method."
