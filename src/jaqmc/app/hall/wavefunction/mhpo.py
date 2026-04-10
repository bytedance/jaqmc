# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Monopole harmonics product orbital (MHPO) wavefunction on the Haldane sphere.

The input features are Cartesian coordinates on the sphere:
``[cos theta, sin theta cos phi, sin theta sin phi]``.
These are fed into a PsiformerBackbone (which appends spin encoding),
followed by monopole harmonic orbital projection and a spherical Jastrow factor.
"""

import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.backbone.psiformer import LayerNormMode, PsiformerBackbone
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction
from jaqmc.wavefunction.output.orbital import SplitChannelDense

from .jastrow import SphericalJastrow

__all__ = ["MHPO"]


class MonopoleOrbitals(nn.Module):
    r"""Monopole harmonic orbital construction.

    Combines learned features with monopole harmonic basis functions
    :math:`u^{Q+m} v^{Q-m}` where :math:`u = \cos(\theta/2) e^{i\phi/2}`
    and :math:`v = \sin(\theta/2) e^{-i\phi/2}`.

    Args:
        Q: Monopole strength.
        nspins: ``(n_up, n_down)`` electron counts.
        ndets: Number of determinants.
    """

    Q: float
    nspins: tuple[int, int]
    ndets: int

    def setup(self) -> None:
        m = np.arange(-self.Q, self.Q + 1)
        self.norm_factor = jnp.array(np.sqrt(ss.comb(2 * self.Q, self.Q - m)))
        features = [int(self.Q * 2) + 1, sum(self.nspins), self.ndets]
        self.orbitals_real = SplitChannelDense(channels=self.nspins, features=features)
        self.orbitals_imag = SplitChannelDense(channels=self.nspins, features=features)

    def __call__(
        self, h_one: jnp.ndarray, theta: jnp.ndarray, phi: jnp.ndarray
    ) -> jnp.ndarray:
        orbitals = self.orbitals_real(h_one) + 1j * self.orbitals_imag(h_one)

        m = jnp.arange(-self.Q, self.Q + 1)
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]
        envelope = self.norm_factor * u ** (self.Q + m) * v ** (self.Q - m)
        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)

        return jnp.moveaxis(orbitals, -1, 0)  # (ndets, nelec, nelec)


class MHPO(Wavefunction[HallData, ComplexWFOutput]):
    r"""Monopole harmonics product orbital ansatz on the Haldane sphere.

    Architecture:
        1. Input: ``(theta, phi)`` to Cartesian features on the sphere
        2. Backbone: :class:`~jaqmc.wavefunction.backbone.psiformer.PsiformerBackbone`
        3. Orbitals: Monopole harmonic projection
        4. Jastrow: Spherical chord-distance Jastrow factor
        5. Output: ``slogdet`` of orbital matrix to complex ``log psi``

    When ``flux_per_elec > 0``, the ansatz implements composite fermion
    mean-field theory by attaching ``flux_per_elec`` flux quanta per electron.
    The effective monopole strength for the orbitals is reduced to
    :math:`Q^* = Q - p(N-1)/2` where :math:`p` is ``flux_per_elec``,
    and a Vandermonde-like Jastrow factor
    :math:`\prod_{i<j}(u_i v_j - u_j v_i)^p` is multiplied in.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        monopole_strength: :math:`Q = \text{flux}/2`.
        flux: Magnetic flux :math:`2Q`.
        ndets: Number of determinants.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        num_layers: Number of Psiformer layers.
        flux_per_elec: Flux quanta attached per electron for composite fermions.
    """

    nspins: tuple[int, int] = runtime_dep()
    monopole_strength: float = runtime_dep()
    flux: int = runtime_dep()
    ndets: int = 1
    num_heads: int = 4
    heads_dim: int = 64
    num_layers: int = 2
    flux_per_elec: int = 0

    def setup(self) -> None:
        reduced_flux = self.flux - self.flux_per_elec * (sum(self.nspins) - 1)
        self.backbone_layer = PsiformerBackbone(
            nspins=self.nspins,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            heads_dim=self.heads_dim,
            mlp_hidden_dims=(),
            layer_norm_mode=LayerNormMode.post,
            input_bias=False,
        )
        self.orbital_layer = MonopoleOrbitals(
            Q=reduced_flux / 2,
            nspins=self.nspins,
            ndets=self.ndets,
        )
        self.jastrow_layer = SphericalJastrow(nspins=self.nspins)

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        theta, phi = electrons[..., 0], electrons[..., 1]

        # Cartesian features on the sphere
        h_one = jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
            ],
            axis=-1,
        )

        # Backbone (appends spin encoding internally)
        h_one = self.backbone_layer(h_one)

        # Monopole harmonic orbitals: (ndets, nelec, nelec)
        orbitals = self.orbital_layer(h_one, theta, phi)

        # Jastrow factor
        jastrow = self.jastrow_layer(electrons)

        # Composite fermion Jastrow attachment
        if self.flux_per_elec > 0:
            u = jnp.cos(theta / 2) * jnp.exp(0.5j * phi)
            v = jnp.sin(theta / 2) * jnp.exp(-0.5j * phi)
            element = u * v[..., None] - u[..., None] * v + jnp.eye(u.shape[0])
            jastrow += jnp.sum(jnp.triu(jnp.log(element), k=1)) * self.flux_per_elec

        orbitals = jnp.exp(jastrow / sum(self.nspins)) * orbitals

        # Complex slogdet
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]
