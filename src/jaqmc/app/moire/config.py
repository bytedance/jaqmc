# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from dataclasses import field
from math import sqrt

import numpy as np

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.units import ONE_HARTREE_IN_MEV, ONE_NM_IN_BOHR

__all__ = ["MoireConfig"]


@configurable_dataclass
class MoireConfig:
    """Configuration for a 2D moire supercell.

    This class holds the user-facing system parameters and derives the
    dimensionless supercell geometry used by the sampler and wavefunction.
    Lattice vectors are in units of ``moire_lattice_constant_nm``; walker
    positions use the same dimensionless coordinate system.

    The field defaults describe a fractional two-thirds-filling system: an
    8-electron spin-polarized ``3x4`` supercell built from a triangular
    primitive cell. Override any ``system.*`` field (via YAML or CLI) to select
    a different system.

    Attributes:
        electron_spins: Number of electrons in the two physical-spin channels
            ``(n_up, n_down)``. This is the fixed physical electron spin, not
            the sampled layer pseudospin (see ``MoireData.spin_coords``).
        nelec: Total number of electrons. Must equal ``sum(electron_spins)``.
        lattice_vectors: Primitive computational lattice vectors.
        moire_lattice_vectors: Physical moire lattice vectors used by moire
            potential and SOC. Defaults to ``lattice_vectors``.
        supercell_matrix: Integer matrix that builds the simulation cell from
            ``lattice_vectors``.
        twist: Twist in simulation-cell reciprocal fractional coordinates.
        effective_mass: Effective mass in units of electron mass.
        dielectric_constant: Relative dielectric constant for Coulomb energy.
        v1_mev: Moire intralayer potential amplitude in meV.
        phi1_deg: Moire potential phase in degrees.
        omega1_mev: Interlayer tunneling amplitude in meV.
        moire_lattice_constant_nm: Physical length scale for dimensionless
            lattice vectors.

    The dimensionless supercell geometry and unit conversions (``scale``,
    ``supercell_lattice``, ``reciprocal_lattice``,
    ``simulation_reciprocal_lattice``, ``moire_lattice_constant_bohr``,
    ``phi1_rad``, ``kinetic_prefactor_mev``, ``coulomb_prefactor_mev``) are
    exposed as derived properties.
    """

    electron_spins: tuple[int, int] = (8, 0)
    nelec: int = 8
    lattice_vectors: list[list[float]] = field(
        default_factory=lambda: [[sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]]
    )
    moire_lattice_vectors: list[list[float]] | None = None
    supercell_matrix: list[list[int]] = field(default_factory=lambda: [[3, 0], [0, 4]])
    twist: list[float] = field(default_factory=lambda: [0.0, 0.0])
    effective_mass: float = 0.62
    dielectric_constant: float = 10.0
    v1_mev: float = 11.2
    phi1_deg: float = -91.0
    omega1_mev: float = -13.3
    moire_lattice_constant_nm: float = 10.084569175744832

    def __post_init__(self) -> None:
        self._validate_electron_counts()
        self._validate_lattices()
        self._validate_supercell()
        self._validate_twist()

        if self.moire_lattice_vectors is None:
            self.moire_lattice_vectors = self.lattice_vectors

    def _validate_electron_counts(self) -> None:
        if len(self.electron_spins) != 2:
            raise ValueError(
                f"electron_spins must have two channels, got {self.electron_spins}."
            )
        if sum(self.electron_spins) != self.nelec:
            raise ValueError(
                "nelec must equal sum(electron_spins), got "
                f"nelec={self.nelec}, electron_spins={self.electron_spins}."
            )

    def _validate_lattices(self) -> None:
        lattice = np.asarray(self.lattice_vectors, dtype=float)
        if lattice.shape != (2, 2):
            raise ValueError(
                f"lattice_vectors must have shape (2, 2), got {lattice.shape}."
            )
        moire_lattice = (
            lattice
            if self.moire_lattice_vectors is None
            else np.asarray(self.moire_lattice_vectors, dtype=float)
        )
        if moire_lattice.shape != (2, 2):
            raise ValueError(
                "moire_lattice_vectors must have shape (2, 2), got "
                f"{moire_lattice.shape}."
            )
        moire_norms = np.linalg.norm(moire_lattice, axis=1)
        if not np.allclose(moire_norms, 1.0, atol=1e-6):
            raise ValueError(
                "moire_lattice_vectors must have unit-norm rows; the physical "
                "length scale is set by moire_lattice_constant_nm. Got row "
                f"norms {moire_norms.tolist()}."
            )

    def _validate_supercell(self) -> None:
        supercell = np.asarray(self.supercell_matrix, dtype=float)
        if supercell.shape != (2, 2):
            raise ValueError(
                f"supercell_matrix must have shape (2, 2), got {supercell.shape}."
            )
        if not np.array_equal(supercell, np.round(supercell)):
            raise ValueError("supercell_matrix must have integer entries.")
        det = round(abs(np.linalg.det(supercell)))
        if det <= 0:
            raise ValueError("supercell_matrix must have non-zero determinant.")

    def _validate_twist(self) -> None:
        twist = np.asarray(self.twist, dtype=float)
        if twist.shape != (2,):
            raise ValueError(f"twist must have shape (2,), got {twist.shape}.")

    @property
    def scale(self) -> int:
        """Number of primitive computational cells in the simulation cell."""
        return abs(round(np.linalg.det(np.asarray(self.supercell_matrix))))

    @property
    def supercell_lattice(self) -> np.ndarray:
        """Simulation-cell lattice vectors derived from the supercell matrix."""
        return np.dot(
            np.asarray(self.supercell_matrix), np.asarray(self.lattice_vectors)
        )

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        """Reciprocal vectors of the primitive computational lattice."""
        return 2.0 * np.pi * np.linalg.inv(np.asarray(self.lattice_vectors)).T

    @property
    def simulation_reciprocal_lattice(self) -> np.ndarray:
        """Reciprocal vectors of the simulation-cell lattice."""
        return 2.0 * np.pi * np.linalg.inv(self.supercell_lattice).T

    @property
    def moire_lattice_constant_bohr(self) -> float:
        """Physical moire lattice constant converted to Bohr."""
        return self.moire_lattice_constant_nm * ONE_NM_IN_BOHR

    @property
    def phi1_rad(self) -> float:
        """Moire potential phase in radians."""
        return float(np.deg2rad(self.phi1_deg))

    @property
    def kinetic_prefactor_mev(self) -> float:
        """Kinetic-energy prefactor in meV for the dimensionless coordinates."""
        return ONE_HARTREE_IN_MEV / (self.moire_lattice_constant_bohr**2)

    @property
    def coulomb_prefactor_mev(self) -> float:
        """Coulomb-energy prefactor in meV for the dimensionless coordinates."""
        return ONE_HARTREE_IN_MEV / (
            self.dielectric_constant * self.moire_lattice_constant_bohr
        )
