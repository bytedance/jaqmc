# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Jastrow factors for wavefunctions on the Haldane sphere."""

from flax import linen as nn
from jax import numpy as jnp


class SphericalJastrow(nn.Module):
    """Spherical chord-distance Jastrow factor.

    Computes electron-electron Jastrow correlations using the chord
    distance on the sphere, with separate parameters for parallel and
    antiparallel spin pairs.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
    """

    nspins: tuple[int, int]

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        nspins = self.nspins
        r_ee = self._chord_distance(electrons)

        r_ees = [
            jnp.split(r, nspins[0:1], axis=1)
            for r in jnp.split(r_ee, nspins[0:1], axis=0)
        ]
        r_ees_parallel = jnp.concatenate(
            [
                r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
                r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
            ]
        )

        if r_ees_parallel.shape[0] > 0:
            alpha_par = self.param("ee_par", nn.initializers.ones, (1,))
            jastrow_ee_par = jnp.sum(
                -(0.25 * alpha_par**2) / (alpha_par + r_ees_parallel)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,))
            jastrow_ee_anti = jnp.sum(
                -(0.5 * alpha_anti**2) / (alpha_anti + r_ees[0][1])
            )
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return jastrow_ee_anti + jastrow_ee_par

    def _chord_distance(self, electrons: jnp.ndarray) -> jnp.ndarray:
        theta, phi = electrons[..., 0], electrons[..., 1]
        cart_e = jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
            ],
            axis=-1,
        )
        cart_ee = cart_e[None] - cart_e[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        return jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)
