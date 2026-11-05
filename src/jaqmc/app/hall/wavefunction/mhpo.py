# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Monopole harmonics product orbital (MHPO) wavefunction on the Haldane sphere."""

from typing import Literal

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax.typing import DTypeLike
from scipy import special as ss

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.geometry.sphere import cartesian_from_spinor
from jaqmc.laplacian.precision import ContractionMode, jacobian_contraction_scope
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput
from jaqmc.wavefunction.output.logdet import LogDet
from jaqmc.wavefunction.output.orbital import SplitChannelDense

from ._backbone import build_hall_backbone
from ._dressing import (
    Dressing,
    DressingConfig,
    reduced_monopole_strength,
)
from ._jastrow import SphericalCoulombJastrow
from .base import HallWavefunction, lll_monopole_harmonics

__all__ = ["MHPO", "DirectMonopoleOrbitals", "FactorizedMonopoleOrbitals"]


class DirectMonopoleOrbitals(nn.Module):
    r"""Direct monopole-harmonic orbital construction.

    Combines learned features with the basis
    :math:`u^{Q+m} v^{Q-m}` where :math:`u = \cos(\theta/2) e^{i\phi/2}`
    and :math:`v = \sin(\theta/2) e^{-i\phi/2}`.

    Args:
        Q: Monopole strength; may be negative for anti-holomorphic shells.
        nspins: ``(n_up, n_down)`` electron counts.
        ndets: Number of determinants.
        dtype: Projection compute dtype. ``None`` follows incoming activations.
    """

    Q: float
    nspins: tuple[int, int]
    ndets: int
    dtype: DTypeLike | None = None

    def setup(self) -> None:
        q_abs = abs(self.Q)
        n_orb = round(2 * q_abs + 1)
        self.norm_factor = jnp.array(np.sqrt(ss.comb(2 * q_abs, np.arange(n_orb))))
        features = [n_orb, sum(self.nspins), self.ndets]
        self.orbitals_real = SplitChannelDense(
            channels=self.nspins, features=features, dtype=self.dtype
        )
        self.orbitals_imag = SplitChannelDense(
            channels=self.nspins, features=features, dtype=self.dtype
        )

    def __call__(
        self, h_one: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray
    ) -> jnp.ndarray:
        """Build orbitals from learned features and monopole spinors.

        Args:
            h_one: Backbone features for every electron.
            u: First spinor coordinate for every electron.
            v: Second spinor coordinate for every electron.

        Returns:
            Determinant-leading orbital matrices.
        """
        orbitals = self.orbitals_real(h_one) + 1j * self.orbitals_imag(h_one)

        q_abs = abs(self.Q)
        envelope = self.norm_factor * lll_monopole_harmonics(u, v, q_abs)
        if self.Q < 0:
            envelope = jnp.conj(envelope)
        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)

        return jnp.moveaxis(orbitals, -1, 0)  # (ndets, nelec, nelec)


class FactorizedMonopoleOrbitals(nn.Module):
    """Factorized monopole-harmonic orbital readout."""

    nspins: tuple[int, int]
    ndets: int
    Q: float
    dtype: DTypeLike | None = None

    @nn.compact
    def __call__(
        self, h_one: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute determinant-leading factorized orbital matrices.

        Returns:
            Orbitals with shape ``(ndets, ..., nelec, nelec)``.
        """
        nelec = h_one.shape[-2]
        feature_dim = h_one.shape[-1]

        q_abs = abs(self.Q)
        m = jnp.arange(-q_abs, q_abs + 1)
        norm_factor = jnp.array(
            np.sqrt(ss.comb(2 * q_abs, q_abs - np.arange(-q_abs, q_abs + 1)))
        )
        envelope = norm_factor * u[..., None] ** (q_abs + m)
        envelope = envelope * v[..., None] ** (q_abs - m)
        if self.Q < 0:
            envelope = jnp.conj(envelope)

        h_splits = jnp.split(h_one, (self.nspins[0],), axis=-2)
        envelope_splits = jnp.split(envelope, (self.nspins[0],), axis=-2)

        orbitals_parts = []
        for i, (h_part, envelope_part) in enumerate(
            zip(h_splits, envelope_splits, strict=True)
        ):
            if h_part.shape[-2] == 0:
                continue

            out_dtype = (
                None if self.dtype is None else jnp.result_type(self.dtype, jnp.float32)
            )
            env_in = jnp.stack((envelope_part.real, envelope_part.imag), axis=-2)
            h_in = h_part
            if self.dtype is not None and out_dtype is not None:
                env_in = env_in.astype(self.dtype)
                h_in = h_part.astype(self.dtype).astype(out_dtype)
            projected_envelope = nn.DenseGeneral(
                features=(self.ndets, feature_dim), use_bias=False, dtype=self.dtype
            )(env_in)
            if out_dtype is not None:
                projected_envelope = projected_envelope.astype(out_dtype)
            x = h_in[..., :, None, :] * (
                projected_envelope[..., 0, :, :] + 1j * projected_envelope[..., 1, :, :]
            )
            stacked_x = jnp.stack((x.real, x.imag), axis=-2)
            if self.dtype is not None:
                stacked_x = stacked_x.astype(self.dtype)
            real_projected = nn.Dense(
                nelec, use_bias=False, dtype=self.dtype, name=f"out_real_{i}"
            )(stacked_x)
            imag_projected = nn.Dense(
                nelec, use_bias=False, dtype=self.dtype, name=f"out_imag_{i}"
            )(stacked_x)
            if out_dtype is not None:
                real_projected = real_projected.astype(out_dtype)
                imag_projected = imag_projected.astype(out_dtype)
            real = real_projected[..., 0, :] + 1j * real_projected[..., 1, :]
            imag = imag_projected[..., 0, :] + 1j * imag_projected[..., 1, :]
            orbitals_parts.append(real + 1j * imag)

        orbitals = jnp.concatenate(orbitals_parts, axis=-3)
        return jnp.moveaxis(orbitals, -2, 0)


class MHPO(HallWavefunction):
    r"""Monopole harmonics product orbital ansatz on the Haldane sphere.

    Architecture:
        1. Input: ``(theta, phi)`` to a monopole spinor and Cartesian point
        2. Backbone: configurable Psiformer or LapNet feature extractor
        3. Orbitals: Direct or factorized monopole-harmonic projection
        4. Jastrow: Coulomb-cusp Jastrow factor on the sphere
        5. Output: ``slogdet`` of orbital matrix to complex ``log psi``

    When ``dressing.degree > 0``, the ansatz attaches that many flux quanta per
    electron using a fixed Laughlin factor or learned orbital pair factors.
    The effective monopole strength for the orbitals is reduced to
    :math:`Q^* = Q - p(N-1)/2` where :math:`p` is ``dressing.degree``,
    independent of the selected dressing.

    Args:
        precision: ``standard`` follows array dtypes. ``fp16_jac`` uses FP32 neural
            evaluation and scaled FP16 Forward Laplacian Jacobian contractions.
            ``fp16_mixed`` runs the backbone, monopole orbital projections, and
            learned dressing in FP16 with the same contractions (retaining an FP32
            residual stream when ``backbone="psiformer"``). Both nonstandard modes
            retain FP32 Jastrow and complex determinant evaluation.
        nspins: ``(n_up, n_down)`` electron counts.
        monopole_strength: :math:`Q = \text{flux}/2`.
        flux: Magnetic flux :math:`2Q`.
        ndets: Number of determinants.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        backbone: Neural backbone implementation.
        num_layers: Number of backbone layers.
        orbital_readout: Monopole-harmonic readout parameterization.
        dressing: Flux dressing configuration (see :class:`DressingConfig`).
        interaction_strength: Coulomb interaction scaling factor for the Jastrow.
        radius: Sphere radius used by the Jastrow cusp (runtime dep, wired by
            the workflow to match the estimators).
    """

    precision: Literal["standard", "fp16_jac", "fp16_mixed"] = "standard"
    nspins: tuple[int, int] = runtime_dep()
    monopole_strength: float = runtime_dep()
    flux: int = runtime_dep()
    interaction_strength: float = runtime_dep()
    radius: float = runtime_dep()
    ndets: int = 1
    backbone: Literal["psiformer", "lapnet"] = "psiformer"
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    orbital_readout: Literal["direct", "factorized"] = "direct"
    dressing: DressingConfig = DressingConfig()

    def setup(self) -> None:
        q_effective = reduced_monopole_strength(
            flux=self.flux,
            degree=self.dressing.degree,
            nelec=sum(self.nspins),
        )
        dtype = {
            "standard": None,
            "fp16_jac": jnp.float32,
            "fp16_mixed": jnp.float16,
        }[self.precision]
        self.backbone_layer = build_hall_backbone(
            backbone=self.backbone,
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
            dtype=dtype,
        )
        self.orbital_layer: nn.Module
        if self.orbital_readout == "direct":
            self.orbital_layer = DirectMonopoleOrbitals(
                Q=q_effective,
                nspins=self.nspins,
                ndets=self.ndets,
                dtype=dtype,
            )
        elif self.orbital_readout == "factorized":
            self.orbital_layer = FactorizedMonopoleOrbitals(
                Q=q_effective,
                nspins=self.nspins,
                ndets=self.ndets,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown orbital readout: {self.orbital_readout!r}")
        self.dressing_layer = Dressing(
            config=self.dressing,
            ndets=self.ndets,
            dtype=dtype,
        )
        self.jastrow_layer = SphericalCoulombJastrow(
            nspins=self.nspins,
            interaction_strength=self.interaction_strength,
            radius=self.radius,
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

    @nn.nowrap
    def evaluate(
        self, params: Params, data: HallData, *, precision: str | None = None
    ) -> ComplexWFOutput:
        """Return structured wavefunction output with an optional mode."""
        module = self if precision is None else self.clone(precision=precision)
        return module.apply(params, data)  # type: ignore

    @nn.nowrap
    def logpsi_from_spinor(
        self,
        params: Params,
        u: jnp.ndarray,
        v: jnp.ndarray,
        *,
        precision: str | None = None,
    ) -> jnp.ndarray:
        """Return log psi from a local monopole spinor with an optional mode."""
        module = self if precision is None else self.clone(precision=precision)
        return module.apply(params, u, v, method=module._logpsi_from_spinor)  # type: ignore

    def _logpsi_from_spinor(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        cartesian = cartesian_from_spinor(u, v)

        with (
            jacobian_contraction_scope(self.contraction_mode),
            jax.default_matmul_precision(
                "bfloat16"
                if self.precision == "fp16_mixed"
                else jax.default_matmul_precision.value
            ),
        ):
            # Backbone (appends spin encoding internally)
            h_one = self.backbone_layer(cartesian)

            # Monopole harmonic orbitals: (ndets, nelec, nelec)
            orbitals = self.orbital_layer(h_one, u, v)
            orbitals = self.dressing_layer(orbitals, u, v, h_one=h_one)

        # Jastrow factor
        jastrow = self.jastrow_layer(cartesian)

        orbitals = jnp.exp(jastrow / sum(self.nspins)) * orbitals

        return self.logdet_layer(orbitals)["logpsi"]
