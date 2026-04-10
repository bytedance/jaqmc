# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import Wavefunction
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet, RealLogDetOutput
from jaqmc.wavefunction.output.orbital import OrbitalProjection

__all__ = ["FermiNetWavefunction"]


class FermiNetWavefunction(Wavefunction[MoleculeData, RealLogDetOutput]):
    """FermiNet ansatz with two-electron feature streams.

    Implements the :class:`~.base.MoleculeWavefunction` protocol.

    FermiNet processes both one-electron (atom-electron) and two-electron
    (electron-electron) features through separate streams with message passing.

    Args:
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        ndets: Number of determinants.
        hidden_dims_single: Hidden dimensions for single-electron stream.
        hidden_dims_double: Hidden dimensions for double-electron stream.
        use_last_layer: If False, skip the double-electron stream update in the
            final layer and return single-electron features directly.
        envelope: Type of envelope function to apply to orbitals.
        orbitals_spin_split: If True, use separate orbital layer and envelope
            parameters for each spin channel. Only effective when both spin
            channels are occupied.

    """

    nspins: tuple[int, int] = runtime_dep()
    ndets: int = 16
    hidden_dims_single: list[int] = field(default_factory=lambda: [256] * 4)
    hidden_dims_double: list[int] = field(default_factory=lambda: [32] * 4)
    use_last_layer: bool = False
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    orbitals_spin_split: bool = True
    # FermiNet always uses full determinant
    full_det: bool = True

    def setup(self) -> None:
        if not self.full_det:
            raise ValueError("FermiNet requires full_det=True.")
        self.feature_layer = MoleculeFeatures()
        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        self.backbone_layer = FermiLayers(
            self.nspins, hidden_dims, use_last_layer=self.use_last_layer
        )

        self.orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )
        self.logdet_layer = LogDet()

    def __call__(self, data: MoleculeData):
        return self.logdet_layer(self.get_orbitals(data))

    def get_orbitals(self, data):
        """Compute orbital matrix from electron and atom positions.

        Args:
            data: MoleculeData containing electrons and atoms.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        embedding = self.feature_layer(data.electrons, data.atoms)
        h_one, _ = self.backbone_layer(
            embedding["ae_features"], embedding["ee_features"]
        )
        orbitals = self.orbital_layer(h_one)
        orbitals = orbitals * self.envelope_layer(
            embedding["ae_vec"], embedding["r_ae"]
        )
        return orbitals

    def logpsi(self, params: Params, data: MoleculeData) -> jnp.ndarray:
        """Evaluate log|psi| given parameters and data.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.

        Returns:
            Scalar log-amplitude of the wavefunction.
        """
        return self.evaluate(params, data)["logpsi"]

    def phase_logpsi(
        self, params: Params, data: MoleculeData
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate (phase, log|psi|) given parameters and data.

        Required for ECP calculations where wavefunction ratios need the phase.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.

        Returns:
            Tuple of (phase, log|psi|) where phase is +1 or -1 for real
            wavefunctions, or a complex unit value for complex wavefunctions.
        """
        out = self.evaluate(params, data)
        return out["sign_logpsi"], out["logpsi"]

    def orbitals(self, params: Params, data: MoleculeData) -> jnp.ndarray:
        """Evaluate orbitals given parameters and data.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        return self.apply(params, data, method=self.get_orbitals)  # type: ignore
