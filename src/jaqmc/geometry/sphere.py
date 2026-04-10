# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Spherical geometry utilities for MCMC sampling on the sphere."""

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey


def _sphere_move(rngs: PRNGKey, x: jnp.ndarray, stddev: float | jnp.ndarray):
    # Rotate a single array of spherical coordinates on the sphere.
    theta, phi = x[..., 0], x[..., 1]
    key_theta, key_phi = jax.random.split(rngs)

    # Generate displacement in rotated frame (north pole)
    theta_prime = jnp.arctan(jax.random.normal(key_theta, shape=theta.shape) * stddev)
    phi_prime = jax.random.uniform(key_phi, phi.shape) * 2 * jnp.pi

    # Convert to Cartesian in rotated frame
    xyz_prime = jnp.stack(
        [
            jnp.sin(theta_prime) * jnp.cos(phi_prime),
            jnp.sin(theta_prime) * jnp.sin(phi_prime),
            jnp.cos(theta_prime),
        ],
        axis=-1,
    )

    # Build rotation matrices to rotate north pole to each electron's position
    one = jnp.ones_like(phi)
    zero = jnp.zeros_like(phi)
    rot_z = jnp.array(
        [
            [jnp.cos(phi), -jnp.sin(phi), zero],
            [jnp.sin(phi), jnp.cos(phi), zero],
            [zero, zero, one],
        ]
    )
    rot_y = jnp.array(
        [
            [jnp.cos(theta), zero, jnp.sin(theta)],
            [zero, one, zero],
            [-jnp.sin(theta), zero, jnp.cos(theta)],
        ]
    )

    # Apply rotation and convert back to spherical coordinates
    x2_xyz = jnp.einsum("ijbn,jkbn,bnk->bni", rot_z, rot_y, xyz_prime)
    x2, y2, z2 = x2_xyz[..., 0], x2_xyz[..., 1], x2_xyz[..., 2]
    new_theta = jnp.arccos(jnp.clip(z2, -1, 1))
    new_phi = jnp.sign(y2) * jnp.arccos(jnp.clip(x2 / jnp.sin(new_theta), -1, 1))
    return jnp.stack([new_theta, new_phi], axis=-1)


def sphere_proposal(rngs: PRNGKey, x, stddev: float | jnp.ndarray):
    """Propose MCMC moves on the sphere, operating on PyTree leaves.

    Applies a rotation-based spherical move to each leaf array in ``x``.
    Each leaf should have shape ``(..., 2)`` where the last axis contains
    ``(theta, phi)`` spherical coordinates.

    Args:
        rngs: Random key.
        x: Current configuration (array or PyTree of arrays).
        stddev: Gaussian width of the angular move.

    Returns:
        New configuration with the same structure as ``x``.
    """
    return jax.tree.map(lambda a: _sphere_move(rngs, a, stddev), x)
