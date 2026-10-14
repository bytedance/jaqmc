# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Jastrow factors for wavefunctions on the Haldane sphere."""

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.optimizer.kfac.tags import register_first_order_parameter


class SphericalCoulombJastrow(nn.Module):
    """Coulomb-cusp Jastrow factor on the Haldane sphere.

    The cusp slope cancels the Coulomb singularity. Exact 2D Coulomb cusp
    coefficients are scaled by ``interaction_strength``.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        interaction_strength: Coulomb interaction scaling factor.
        radius: Sphere radius.
    """

    nspins: tuple[int, int]
    interaction_strength: float
    radius: float

    @nn.compact
    def __call__(self, cartesian: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Jastrow factor from Cartesian sphere coordinates.

        Args:
            cartesian: Unit-sphere Cartesian electron positions.

        Returns:
            The scalar Jastrow contribution.
        """
        nspins = self.nspins
        r_ee = self._chord_distance(cartesian) * self.radius

        alpha_init = nn.initializers.ones

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
            alpha_par = self.param("ee_par", alpha_init, (1,), jnp.float32)
            contribution = -(1 / 3) * jnp.sum(
                alpha_par**2 / (alpha_par + r_ees_parallel)
            )
            jastrow_ee_par = register_first_order_parameter(contribution, alpha_par)
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].size > 0:
            alpha_anti = self.param("ee_anti", alpha_init, (1,), jnp.float32)
            contribution = -jnp.sum(alpha_anti**2 / (alpha_anti + r_ees[0][1]))
            jastrow_ee_anti = register_first_order_parameter(contribution, alpha_anti)
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return (jastrow_ee_anti + jastrow_ee_par) * self.interaction_strength

    def _chord_distance(self, cartesian: jnp.ndarray) -> jnp.ndarray:
        """Return all unit-sphere pair chord distances from Cartesian points."""
        cart_ee = cartesian[None] - cartesian[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        return jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)
