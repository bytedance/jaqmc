# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal, Protocol

import numpy as np
import serde

from jaqmc.utils.atomic import Atom, AtomInitialization
from jaqmc.utils.atomic.atomic_system import AtomicSystemConfig
from jaqmc.utils.atomic.pretrain import PretrainReferenceConfig
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

__all__ = ["SolidAtomConfig", "SolidConfig", "SolidPretrainReferenceConfig"]


@configurable_dataclass(kw_only=False)
class SolidAtomConfig:
    """Input configuration for one primitive-cell atom in fractional coordinates."""

    symbol: str
    frac_coords: list[float]
    charge: int | None = None
    initialization: AtomInitialization = field(default_factory=AtomInitialization)

    def __post_init__(self):
        if len(self.frac_coords) != 3:
            raise ValueError(
                "Expected three-dimensional fractional coordinates. "
                f"Got {len(self.frac_coords)} in {self}."
            )
        if any(coord < 0.0 or coord >= 1.0 for coord in self.frac_coords):
            raise ValueError(
                "Fractional coordinates must satisfy 0 <= coord < 1 for solids. "
                f"Got {self.frac_coords!r}."
            )


class LatticeSpec(Protocol):
    def to_array(self) -> np.ndarray: ...


@configurable_dataclass
class LatticeDirect(LatticeSpec):
    a: tuple[float | int, float | int, float | int]
    b: tuple[float | int, float | int, float | int]
    c: tuple[float | int, float | int, float | int]

    def to_array(self) -> np.ndarray:
        return np.asarray([list(self.a), list(self.b), list(self.c)])


@configurable_dataclass
class LatticeParams(LatticeSpec):
    a: float | int
    b: float | int
    c: float | int
    alpha: float | int = 90
    beta: float | int = 90
    gamma: float | int = 90

    def to_array(self) -> np.ndarray:
        # Handle orthorhombic cells separately to avoid rounding errors
        cos_alpha = _cosd_snap_90(self.alpha)
        cos_beta = _cosd_snap_90(self.beta)
        cos_gamma = _cosd_snap_90(self.gamma)
        sin_gamma = _sind_snap_90(self.gamma)

        cx = cos_beta
        cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz_sqr = 1.0 - cx**2 - cy**2
        if cz_sqr < 0 or np.isclose(cz_sqr, 0):
            raise ValueError("Invalid cell geometry: c vector collapse with a and b.")

        return np.array(
            [
                [self.a, 0.0, 0.0],
                [self.b * cos_gamma, self.b * sin_gamma, 0.0],
                [self.c * cx, self.c * cy, self.c * np.sqrt(cz_sqr)],
            ],
        )


@configurable_dataclass
class SolidConfig(AtomicSystemConfig):
    """Configuration for solid-state/periodic systems.

    This class holds the configuration for solid-state systems, including
    lattice vectors, supercell matrix, and twist vectors. It inherits from
    :class:`~jaqmc.utils.atomic.atomic_system.AtomicSystemConfig`.

    Args:
        atoms: Primitive-cell atom basis in fractional coordinates.
        lattice: Primitive-cell lattice, given either as direct vectors
            (`a`, `b`, `c`) or as cell parameters (`a`, `b`, `c`,
            `alpha`, `beta`, `gamma`).
        supercell_matrix: Optional 3x3 integer matrix defining a supercell.
        twist: Twist vector in fractional reciprocal coordinates.
        unit: Length unit applied to lattice vectors.
    """

    lattice: LatticeDirect | LatticeParams = field(
        default_factory=lambda: LatticeParams(a=2, b=2, c=2)
    )
    unit: LengthUnit = LengthUnit.bohr
    atom_configs: list[SolidAtomConfig] = serde.field(
        rename="atoms",
        default_factory=lambda: [
            SolidAtomConfig(symbol="H", frac_coords=[0, 0, 0]),
            SolidAtomConfig(symbol="H", frac_coords=[0.5, 0, 0]),
        ],
    )
    supercell_matrix: list[list[int]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    twist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        super().__post_init__()

        supercell_arr = np.asarray(self.supercell_matrix)
        if supercell_arr.shape != (3, 3):
            raise ValueError(
                f"supercell_matrix must be a 3x3 matrix. "
                f"Got shape {supercell_arr.shape}."
            )
        if not np.allclose(supercell_arr, np.round(supercell_arr)):
            raise ValueError("supercell_matrix must have integer entries.")

    @property
    def lattice_vectors(self) -> np.ndarray:
        if self.unit == LengthUnit.bohr:
            return self.lattice.to_array()
        return self.lattice.to_array() * ONE_ANGSTROM_IN_BOHR

    @property
    def supercell_lattice(self) -> np.ndarray:
        return np.dot(np.asarray(self.supercell_matrix), self.lattice_vectors)

    @property
    def scale(self) -> int:
        return abs(round(np.linalg.det(self.supercell_matrix)))

    @property
    def atoms(self) -> list[Atom]:
        return [
            Atom(
                symbol=atom.symbol,
                coords=(np.asarray(atom.frac_coords) @ self.lattice_vectors).tolist(),
                charge=atom.charge,
                pp=self.pp.get(atom.symbol) if isinstance(self.pp, dict) else self.pp,
            )
            for atom in self.atom_configs
        ]

    @property
    def per_atom_init(self) -> list[AtomInitialization]:
        return [atom.initialization for atom in self.atom_configs]


@configurable_dataclass
class SolidPretrainReferenceConfig(PretrainReferenceConfig):
    method: Literal["KRHF", "KUHF"] = "KUHF"
    "Variants of Hartree-Fock method."


def _cosd_snap_90(angle, *, atol=1e-12):
    if np.isclose(abs(angle), 90.0, atol=atol, rtol=0.0):
        return 0.0
    return np.cos(np.deg2rad(angle))


def _sind_snap_90(angle, *, atol=1e-12):
    if np.isclose(abs(angle), 90.0, atol=atol, rtol=0.0):
        return float(np.copysign(1.0, angle))
    return np.sin(np.deg2rad(angle))
