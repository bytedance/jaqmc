# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""LapNet wavefunction implementation for molecules."""

from enum import StrEnum
from typing import Literal, cast

import jax
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.base import MoleculeWavefunctionModule
from jaqmc.geometry import obc
from jaqmc.laplacian.precision import ContractionMode, jacobian_contraction_scope
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.backbone.lapnet import LapNetBackbone
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.jastrow import SimpleEEJastrow
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet, RealLogDetOutput
from jaqmc.wavefunction.output.orbital import OrbitalProjection

__all__ = ["LapNetWavefunction"]


class JastrowType(StrEnum):
    """Available Jastrow factor types for LapNet wavefunction.

    Attributes:
        NONE: Disable the Jastrow factor.
        SIMPLE_EE: Enable the built-in electron-electron Jastrow factor.
    """

    NONE = "none"
    SIMPLE_EE = "simple_ee"


class LapNetWavefunction(MoleculeWavefunctionModule):
    """LapNet ansatz with the original two-stream cross-attention backbone.

    Implements the :class:`~.base.MoleculeWavefunction` protocol.

    Args:
        precision: ``standard`` follows array dtypes. ``fp16_jac`` uses FP32 neural
            evaluation and scaled FP16 Forward Laplacian Jacobian contractions.
            ``fp16_mixed`` runs the LapNet backbone and orbital projections in FP16
            with the same contractions (enable ``use_layernorm=True`` to keep FP16
            activations well-scaled). Both nonstandard modes retain FP32 envelope,
            Jastrow, and determinant evaluation.
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        ndets: Number of determinants.
        num_layers: Number of LapNet network layers.
        num_heads: Number of cross-attention heads.
        heads_dim: Dimension of each attention head. The hidden width is
            ``num_heads * heads_dim``.
        use_layernorm: Whether to apply LayerNorm in LapNet blocks.
        jastrow: Jastrow factor type.
        use_input_bias: Whether to use bias in the input projection layer.
        use_backbone_bias: Whether to use bias in LapNet backbone projections.
        num_local_updates: Number of local residual updates on the individual
            stream between attention blocks.
        envelope: Envelope type for orbital decay at infinity.
        use_orbital_bias: Whether to use bias in the orbital projection layer.
        rescale: If True, use log-scaled atom-electron input features.
        jastrow_alpha_init: Initial value for Jastrow alpha parameters.
    """

    precision: Literal["standard", "fp16_jac", "fp16_mixed"] = "standard"
    nspins: tuple[int, int] = runtime_dep()
    ndets: int = 16
    num_layers: int = 4
    num_heads: int = 4
    heads_dim: int = 64
    use_layernorm: bool = False
    jastrow: JastrowType = JastrowType.SIMPLE_EE
    use_input_bias: bool = True
    use_backbone_bias: bool = True
    num_local_updates: int = 2
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    use_orbital_bias: bool = False
    rescale: bool = True
    jastrow_alpha_init: float = 1.0
    full_det: bool = True

    def setup(self) -> None:
        if not self.full_det:
            raise ValueError("LapNet requires full_det=True.")

        self.feature_layer = MoleculeFeatures(rescale=self.rescale)
        dtype = {
            "standard": None,
            "fp16_jac": jnp.float32,
            "fp16_mixed": jnp.float16,
        }[self.precision]
        self.backbone_layer = LapNetBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
            use_layernorm=self.use_layernorm,
            use_input_bias=self.use_input_bias,
            use_backbone_bias=self.use_backbone_bias,
            num_local_updates=self.num_local_updates,
            dtype=dtype,
        )
        self.orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            use_bias=self.use_orbital_bias,
            dtype=dtype,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=True,
        )
        self.logdet_layer = LogDet(
            dtype=None if self.precision == "standard" else jnp.float32
        )

        self.jastrow_layer: SimpleEEJastrow | None
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

        Args:
            data: MoleculeData containing electron and atom positions.

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
            h_one = self.backbone_layer(embedding["ae_features"])
            orbitals = self.orbital_layer(h_one)
        return orbitals * self.envelope_layer(embedding["ae_vec"], embedding["r_ae"])
