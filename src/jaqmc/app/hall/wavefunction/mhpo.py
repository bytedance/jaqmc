# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Monopole harmonics product orbital (MHPO) wavefunction on the Haldane sphere."""

from typing import Literal

import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.geometry.sphere import cartesian_from_spinor
from jaqmc.utils.wiring import runtime_dep
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
    """

    Q: float
    nspins: tuple[int, int]
    ndets: int

    def setup(self) -> None:
        q_abs = abs(self.Q)
        n_orb = round(2 * q_abs + 1)
        self.norm_factor = jnp.array(np.sqrt(ss.comb(2 * q_abs, np.arange(n_orb))))
        features = [n_orb, sum(self.nspins), self.ndets]
        self.orbitals_real = SplitChannelDense(channels=self.nspins, features=features)
        self.orbitals_imag = SplitChannelDense(channels=self.nspins, features=features)

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

            projected_envelope = nn.DenseGeneral(
                features=(self.ndets, feature_dim), use_bias=False
            )(jnp.stack((envelope_part.real, envelope_part.imag), axis=-2))
            x = h_part[..., :, None, :] * (
                projected_envelope[..., 0, :, :] + 1j * projected_envelope[..., 1, :, :]
            )
            stacked_x = jnp.stack((x.real, x.imag), axis=-2)
            real_projected = nn.Dense(nelec, use_bias=False, name=f"out_real_{i}")(
                stacked_x
            )
            imag_projected = nn.Dense(nelec, use_bias=False, name=f"out_imag_{i}")(
                stacked_x
            )
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
        self.backbone_layer = build_hall_backbone(
            backbone=self.backbone,
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
        )
        self.orbital_layer: nn.Module
        if self.orbital_readout == "direct":
            self.orbital_layer = DirectMonopoleOrbitals(
                Q=q_effective,
                nspins=self.nspins,
                ndets=self.ndets,
            )
        elif self.orbital_readout == "factorized":
            self.orbital_layer = FactorizedMonopoleOrbitals(
                Q=q_effective,
                nspins=self.nspins,
                ndets=self.ndets,
            )
        else:
            raise ValueError(f"Unknown orbital readout: {self.orbital_readout!r}")
        self.dressing_layer = Dressing(
            config=self.dressing,
            ndets=self.ndets,
        )
        self.jastrow_layer = SphericalCoulombJastrow(
            nspins=self.nspins,
            interaction_strength=self.interaction_strength,
            radius=self.radius,
        )

    def _logpsi_from_spinor(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        cartesian = cartesian_from_spinor(u, v)

        # Backbone (appends spin encoding internally)
        h_one = self.backbone_layer(cartesian)

        # Monopole harmonic orbitals: (ndets, nelec, nelec)
        orbitals = self.orbital_layer(h_one, u, v)
        orbitals = self.dressing_layer(orbitals, u, v, h_one=h_one)

        # Jastrow factor
        jastrow = self.jastrow_layer(cartesian)

        orbitals = jnp.exp(jastrow / sum(self.nspins)) * orbitals

        # Complex slogdet
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax
