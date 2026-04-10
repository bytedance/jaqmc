# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Psiformer wavefunction implementation."""

from dataclasses import field
from enum import StrEnum

from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.geometry import obc
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import Wavefunction
from jaqmc.wavefunction.backbone.psiformer import LayerNormMode, PsiformerBackbone
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.jastrow import SimpleEEJastrow
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet, RealLogDetOutput
from jaqmc.wavefunction.output.orbital import OrbitalProjection

__all__ = ["PsiformerWavefunction"]


class JastrowType(StrEnum):
    """Available Jastrow factor types for Psiformer wavefunction.

    Attributes:
        NONE: Disable the Jastrow factor.
        SIMPLE_EE: Enable the built-in electron-electron Jastrow factor.
    """

    NONE = "none"
    SIMPLE_EE = "simple_ee"


class PsiformerWavefunction(Wavefunction[MoleculeData, RealLogDetOutput]):
    """Psiformer ansatz with self-attention mechanism.

    Implements the :class:`~.base.MoleculeWavefunction` protocol.

    Args:
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        ndets: Number of determinants.
        num_layers: Number of Psiformer layers.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        mlp_hidden_dims: Hidden dimensions for MLP blocks.
        layer_norm_mode: LayerNorm application mode. Options:

            - :attr:`LayerNormMode.pre`: Apply LayerNorm before attention/MLP
              blocks (Pre-LN, matches internal_ferminet). Default.
            - :attr:`LayerNormMode.post`: Apply LayerNorm after attention/MLP
              blocks (Post-LN, matches public FermiNet).
            - :attr:`LayerNormMode.null`: No LayerNorm applied.

        jastrow: Jastrow factor type.
        with_bias: Whether to use bias in attention QKV projections.
        input_bias: Whether to use bias in the input projection layer.
        envelope: Envelope type for orbital decay at infinity.
        orbitals_spin_split: If True, use separate orbital layer and envelope
            parameters for each spin channel. Only effective when both spin
            channels are occupied.
        bias_orbitals: If True, include bias in the orbital projection layer.
        jastrow_alpha_init: Initial value for Jastrow alpha parameters.
        rescale: If True, use log-scaled input features (log(1+r)) instead of
            linear features. Log-scaling provides better numerical stability
            for large distances.

    """

    nspins: tuple[int, int] = runtime_dep()
    ndets: int = 16
    num_layers: int = 4
    num_heads: int = 4
    heads_dim: int = 64
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [256])
    layer_norm_mode: LayerNormMode = LayerNormMode.pre
    jastrow: JastrowType = JastrowType.SIMPLE_EE
    with_bias: bool = True
    input_bias: bool = True
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    orbitals_spin_split: bool = True
    bias_orbitals: bool = False
    rescale: bool = True
    jastrow_alpha_init: float = 1.0
    # Psiformer always uses full determinant
    full_det: bool = True

    def setup(self) -> None:
        if not self.full_det:
            raise ValueError("Psiformer requires full_det=True.")
        self.feature_layer = MoleculeFeatures(rescale=self.rescale)
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
            mlp_hidden_dims=tuple(self.mlp_hidden_dims),
            layer_norm_mode=self.layer_norm_mode,
            with_bias=self.with_bias,
            input_bias=self.input_bias,
        )

        self.orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=self.bias_orbitals,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )
        self.logdet_layer = LogDet()

        self.jastrow_layer: SimpleEEJastrow | None
        # Initialize Jastrow factor based on type
        match self.jastrow:
            case JastrowType.SIMPLE_EE:
                self.jastrow_layer = SimpleEEJastrow(
                    nspins=self.nspins,
                    alpha_init=self.jastrow_alpha_init,
                )
            case JastrowType.NONE:
                self.jastrow_layer = None
            case _:
                raise ValueError(
                    f"Invalid jastrow: {self.jastrow!r}. "
                    f"Must be one of: {[e.value for e in JastrowType]}"
                )

    def __call__(self, data: MoleculeData):
        orbitals = self.get_orbitals(data)
        output = self.logdet_layer(orbitals)

        # Add Jastrow in log-space (numerically stable)
        if self.jastrow_layer is not None:
            _, r_ee = obc.pair_displacements_within(data.electrons)
            jastrow_value = self.jastrow_layer(r_ee)
            output = {**output, "logpsi": output["logpsi"] + jastrow_value}

        return output

    def get_orbitals(self, data):
        """Compute orbital matrix from electron and atom positions.

        Returns pure orbitals with envelope applied but without Jastrow factor.
        The Jastrow factor is added in log-space during wavefunction evaluation.

        Args:
            data: MoleculeData containing electrons and atoms.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        embedding = self.feature_layer(data.electrons, data.atoms)
        # Psiformer only uses ae features (ee interactions via attention)
        h_one = self.backbone_layer(embedding["ae_features"])
        orbitals = self.orbital_layer(h_one)

        # Apply envelope
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
