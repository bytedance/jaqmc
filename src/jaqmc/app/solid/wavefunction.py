# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

from jax import numpy as jnp

from jaqmc.array_types import Params
from jaqmc.geometry.pbc import DistanceType, SymmetryType
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import Wavefunction
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import SolidFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import ComplexLogDetOutput, LogDet
from jaqmc.wavefunction.output.orbital import OrbitalProjection

from .data import SolidData

__all__ = ["SolidWavefunction"]


class SolidWavefunction(Wavefunction[SolidData, ComplexLogDetOutput]):
    """Wavefunction ansatz for solid state systems (periodic boundary conditions).

    Args:
        nspins: Tuple of (n_up, n_down) electrons.
        ndets: Number of determinants.
        simulation_lattice: Lattice vectors of the simulation cell.
        primitive_lattice: Lattice vectors of the primitive cell.
        klist: K-point list for Bloch phases.
        hidden_dims_single: Hidden dimensions for single-electron streams.
        hidden_dims_double: Hidden dimensions for pairwise streams.
        distance_type: Method to compute distances (e.g., 'nu' for nearest image).
        envelope_type: Type of envelope function to use.
        sym_type: Symmetry type for features.
        orbitals_spin_split: If True, use separate orbital layer and envelope
            parameters for each spin channel. Only effective when both spin
            channels are occupied.
        full_det: Whether to use full determinants (default: True).
    """

    nspins: tuple[int, int] = runtime_dep()
    klist: jnp.ndarray = runtime_dep()
    simulation_lattice: jnp.ndarray = runtime_dep()
    primitive_lattice: jnp.ndarray = runtime_dep()
    hidden_dims_single: list[int] = field(default_factory=lambda: [256] * 4)
    hidden_dims_double: list[int] = field(default_factory=lambda: [32] * 4)
    ndets: int = 16
    distance_type: DistanceType = DistanceType.nu
    envelope_type: EnvelopeType = EnvelopeType.abs_isotropic
    sym_type: SymmetryType = SymmetryType.minimal
    orbitals_spin_split: bool = True
    full_det: bool = True

    def setup(self) -> None:
        self.feature_layer = SolidFeatures(
            simulation_lattice=self.simulation_lattice,
            primitive_lattice=self.primitive_lattice,
            distance_type=self.distance_type,
            sym_type=self.sym_type,
        )

        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        self.backbone_layer = FermiLayers(self.nspins, hidden_dims)

        self.real_orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
        )
        self.imag_orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
        )

        self.envelope_layer = Envelope(
            envelope_type=self.envelope_type,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )

        self.logdet_layer = LogDet()

    def get_orbitals(self, data: SolidData) -> jnp.ndarray:
        """Computes the orbital matrix for the given data.

        Args:
            data: The input data containing electron positions and lattice info.

        Returns:
            Complex orbital matrix with shape (ndets, nelec_total, nelec_total).
        """
        embedding = self.feature_layer(data.electrons, data.primitive_atoms)
        h_one, _ = self.backbone_layer(
            embedding["ae_features"], embedding["ee_features"]
        )

        # Combine as a complex orbital matrix (ndets, nelec_total, nelec_total)
        orbital = self.real_orbital_layer(h_one) + 1j * self.imag_orbital_layer(h_one)
        envelope_val = self.envelope_layer(embedding["ae_vec"], embedding["r_ae"])
        orbital = orbital * envelope_val

        phases = jnp.exp(1j * jnp.dot(data.electrons, self.klist.T))
        # orbital shape: (ndets, i(electron), j(orbital))
        orbital = orbital * phases[None, :, :]

        return orbital

    def __call__(self, data: SolidData):
        """Evaluates the wavefunction for the given data.

        Args:
            data: The input data containing electron positions and lattice info.

        Returns:
            A dictionary containing the log of the wavefunction amplitude ('logpsi')
            and other outputs (sign, etc.) from LogDet.
        """
        return self.logdet_layer(self.get_orbitals(data))

    def orbitals(self, params: Params, data: SolidData) -> jnp.ndarray:
        """Evaluates the orbital matrix for the given parameters and data.

        Args:
            params: The wavefunction parameters.
            data: The input data containing electron positions and lattice info.

        Returns:
            Complex orbital matrix with shape (ndets, nelec_total, nelec_total).
        """
        return self.apply(params, data, method=self.get_orbitals)  # type: ignore

    def logpsi(self, params, data):
        return self.evaluate(params, data)["logpsi"]

    def phase_logpsi(
        self, params: Params, data: SolidData
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logpsi = self.logpsi(params, data)
        return jnp.exp(1j * logpsi.imag), logpsi.real
