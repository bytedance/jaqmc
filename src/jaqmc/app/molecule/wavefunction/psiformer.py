# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Psiformer wavefunction implementation."""

from dataclasses import field
from enum import StrEnum
from typing import Literal, cast

import jax
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.base import MoleculeWavefunctionModule
from jaqmc.geometry import obc
from jaqmc.laplacian.precision import ContractionMode, jacobian_contraction_scope
from jaqmc.utils.wiring import runtime_dep
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


class PsiformerWavefunction(MoleculeWavefunctionModule):
    """Psiformer ansatz with self-attention mechanism.

    Implements the :class:`~.base.MoleculeWavefunction` protocol.

    Args:
        precision: ``standard`` follows array dtypes. ``fp16_jac`` uses FP32 neural
            evaluation and scaled FP16 Forward Laplacian Jacobian contractions.
            ``fp16_mixed`` runs attention, MLP, and orbital projections in FP16 with
            the same contractions while retaining an FP32 residual stream and FP32
            LayerNorm reductions. Both nonstandard modes retain FP32 envelope,
            Jastrow, and determinant evaluation.
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        ndets: Number of determinants.
        num_layers: Number of Psiformer layers.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        mlp_hidden_dims: Hidden dimensions for MLP blocks.
        layer_norm_mode: LayerNorm application mode (:attr:`LayerNormMode.pre`
            before attention/MLP blocks, :attr:`LayerNormMode.post` after, or
            :attr:`LayerNormMode.null` to disable).
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
        full_det: Whether to use full determinants (must be ``True``).
    """

    precision: Literal["standard", "fp16_jac", "fp16_mixed"] = "standard"
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
        dtype = {
            "standard": None,
            "fp16_jac": jnp.float32,
            "fp16_mixed": jnp.float16,
        }[self.precision]
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
            mlp_hidden_dims=tuple(self.mlp_hidden_dims),
            layer_norm_mode=self.layer_norm_mode,
            with_bias=self.with_bias,
            input_bias=self.input_bias,
            dtype=dtype,
        )

        self.orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=self.bias_orbitals,
            dtype=dtype,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )
        self.logdet_layer = LogDet(
            dtype=None if self.precision == "standard" else jnp.float32
        )

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

    @property
    def contraction_mode(self) -> ContractionMode:
        """Forward Laplacian contraction mode for this precision strategy."""
        return (
            ContractionMode.SCALED_FP16
            if self.precision != "standard"
            else ContractionMode.STANDARD
        )

    def __call__(self, data: MoleculeData) -> RealLogDetOutput:
        output = cast(RealLogDetOutput, self.logdet_layer(self.get_orbitals(data)))
        if self.jastrow_layer is not None:
            _, r_ee = obc.pair_displacements_within(data.electrons)
            output = {
                **output,
                "logpsi": output["logpsi"] + self.jastrow_layer(r_ee),
            }
        return output

    def get_orbitals(self, data: MoleculeData) -> jnp.ndarray:
        """Compute orbital matrix from electron and atom positions.

        Returns pure orbitals with envelope applied but without Jastrow factor.
        The Jastrow factor is added in log-space during wavefunction evaluation.

        Args:
            data: MoleculeData containing electrons and atoms.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        embedding = self.feature_layer(data.electrons, data.atoms)
        with (
            jacobian_contraction_scope(self.contraction_mode),
            jax.default_matmul_precision(
                "bfloat16"
                if self.precision == "fp16_mixed"
                else jax.default_matmul_precision.value
            ),
        ):
            # Psiformer only uses ae features (ee interactions via attention)
            h_one = self.backbone_layer(embedding["ae_features"])
            orbitals = self.orbital_layer(h_one)

        # Apply envelope
        orbitals = orbitals * self.envelope_layer(
            embedding["ae_vec"], embedding["r_ae"]
        )

        return orbitals
