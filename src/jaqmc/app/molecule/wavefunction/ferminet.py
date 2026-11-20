# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal, cast

import jax
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.base import MoleculeWavefunctionModule
from jaqmc.laplacian.precision import ContractionMode, jacobian_contraction_scope
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet, RealLogDetOutput
from jaqmc.wavefunction.output.orbital import OrbitalProjection

__all__ = ["FermiNetWavefunction"]


class FermiNetWavefunction(MoleculeWavefunctionModule):
    """FermiNet ansatz with two-electron feature streams.

    Implements the :class:`~.base.MoleculeWavefunction` protocol.

    FermiNet processes both one-electron (atom-electron) and two-electron
    (electron-electron) features through separate streams with message passing.

    Args:
        precision: ``standard`` follows array dtypes. ``fp16_jac`` uses FP32 neural
            evaluation and scaled FP16 Forward Laplacian Jacobian contractions.
            ``fp16_mixed`` runs the FermiLayers backbone and orbital projections in
            FP16 (with ``bfloat16`` default matmul precision) alongside scaled FP16
            Jacobian contractions. Both nonstandard modes retain FP32 envelope and
            determinant evaluation.
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

    precision: Literal["standard", "fp16_jac", "fp16_mixed"] = "standard"
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
        dtype = {
            "standard": None,
            "fp16_jac": jnp.float32,
            "fp16_mixed": jnp.float16,
        }[self.precision]
        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        self.backbone_layer = FermiLayers(
            self.nspins,
            hidden_dims,
            use_last_layer=self.use_last_layer,
            dtype=dtype,
        )

        self.orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
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

    @property
    def contraction_mode(self) -> ContractionMode:
        """Forward Laplacian contraction mode for this precision strategy."""
        return (
            ContractionMode.SCALED_FP16
            if self.precision != "standard"
            else ContractionMode.STANDARD
        )

    def __call__(self, data: MoleculeData) -> RealLogDetOutput:
        return cast(RealLogDetOutput, self.logdet_layer(self.get_orbitals(data)))

    def get_orbitals(self, data: MoleculeData) -> jnp.ndarray:
        """Compute orbital matrix from electron and atom positions.

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
            h_one, _ = self.backbone_layer(
                embedding["ae_features"], embedding["ee_features"]
            )
            orbitals = self.orbital_layer(h_one)
        orbitals = orbitals * self.envelope_layer(
            embedding["ae_vec"], embedding["r_ae"]
        )
        return orbitals
