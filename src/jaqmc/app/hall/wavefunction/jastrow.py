# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
    def __call__(self, cartesian: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Jastrow factor from Cartesian sphere coordinates.

        Args:
            cartesian: Unit-sphere Cartesian electron positions.

        Returns:
            The scalar Jastrow contribution.
        """
        nspins = self.nspins
        r_ee = self._chord_distance(cartesian)

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
            alpha_par = self.param("ee_par", nn.initializers.ones, (1,), jnp.float32)
            jastrow_ee_par = jnp.sum(
                -(0.25 * alpha_par**2) / (alpha_par + r_ees_parallel)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,), jnp.float32)
            jastrow_ee_anti = jnp.sum(
                -(0.5 * alpha_anti**2) / (alpha_anti + r_ees[0][1])
            )
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return jastrow_ee_anti + jastrow_ee_par

    def _chord_distance(self, cartesian: jnp.ndarray) -> jnp.ndarray:
        """Return all pair chord distances from Cartesian sphere coordinates."""
        cart_ee = cartesian[None] - cartesian[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        return jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)
