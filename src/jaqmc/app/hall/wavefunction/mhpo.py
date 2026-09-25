# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Monopole harmonics product orbital (MHPO) wavefunction on the Haldane sphere.

Cartesian coordinates on the unit sphere feed a configurable neural backbone
(which appends spin encoding), followed by monopole harmonic orbital projection
and a spherical Jastrow factor.
"""

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
from .base import HallWavefunction, lll_monopole_harmonics
from .jastrow import SphericalJastrow

__all__ = ["MHPO"]


class MonopoleOrbitals(nn.Module):
    r"""Monopole harmonic orbital construction.

    Combines learned features with monopole harmonic basis functions
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


class MHPO(HallWavefunction):
    r"""Monopole harmonics product orbital ansatz on the Haldane sphere.

    Architecture:
        1. Input: ``(theta, phi)`` to a monopole spinor and Cartesian point
        2. Backbone: configurable Psiformer or LapNet feature extractor
        3. Orbitals: Monopole harmonic projection
        4. Jastrow: Spherical chord-distance Jastrow factor
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
        dressing: Fixed or learned flux dressing configuration. ``type`` selects
            Laughlin or full matrix pair factors; ``conditioning`` selects
            constant or backbone-conditioned factors; and ``orbital_dependent``
            and ``det_dependent`` control whether each orbital and each
            determinant has its own pair factor.
    """

    nspins: tuple[int, int] = runtime_dep()
    monopole_strength: float = runtime_dep()
    flux: int = runtime_dep()
    ndets: int = 1
    backbone: Literal["psiformer", "lapnet"] = "psiformer"
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
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
        self.orbital_layer = MonopoleOrbitals(
            Q=q_effective,
            nspins=self.nspins,
            ndets=self.ndets,
        )
        self.dressing_layer = Dressing(
            config=self.dressing,
            ndets=self.ndets,
        )
        self.jastrow_layer = SphericalJastrow(nspins=self.nspins)

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
