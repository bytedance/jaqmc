# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Spherical geometry utilities for the Haldane sphere.

Covers coordinate charts, monopole spinors, the monopole connection, and MCMC
proposals for the Haldane sphere.
"""

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey


def cartesian_from_spinor(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Return the gauge-neutral unit-sphere point of a nonzero spinor.

    Args:
        u: First spinor coordinate.
        v: Second spinor coordinate.

    Returns:
        Cartesian coordinates with trailing ``(x, y, z)`` axis.
    """
    # Products with the complex conjugates are used instead of ``abs(...) ** 2`` so
    # second derivatives remain well-defined when a component vanishes at a pole.
    u_abs2 = jnp.real(jnp.conj(u) * u)
    v_abs2 = jnp.real(jnp.conj(v) * v)
    norm_squared = u_abs2 + v_abs2
    uv = jnp.conj(u) * v
    return (
        jnp.stack([2 * jnp.real(uv), -2 * jnp.imag(uv), u_abs2 - v_abs2], axis=-1)
        / norm_squared[..., None]
    )


def spinor_coordinates_from_angles(
    electrons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a symmetric-gauge monopole spinor.

    Args:
        electrons: Spherical coordinates with trailing ``(theta, phi)`` axis.

    Returns:
        The symmetric-gauge monopole spinor ``(u, v)``.
    """
    theta, phi = electrons[..., 0], electrons[..., 1]
    u = jnp.cos(theta / 2) * jnp.exp(0.5j * phi)
    v = jnp.sin(theta / 2) * jnp.exp(-0.5j * phi)
    return u, v


def stereographic_coordinates(
    electrons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Nearest-pole stereographic ``(x, y)`` and chart flag per electron.

    North (``theta <= pi/2``): ``r = tan(theta/2)``.
    South: ``r = tan((pi - theta)/2)``.

    Returns:
        ``(plane, is_north)`` with trailing plane axis ``(x, y)``.
    """
    theta, phi = electrons[..., 0], electrons[..., 1]
    is_north = theta <= jnp.pi / 2
    radial_coordinate = jnp.where(
        is_north,
        jnp.tan(theta / 2),
        jnp.tan((jnp.pi - theta) / 2),
    )
    plane = jnp.stack(
        [
            radial_coordinate * jnp.cos(phi),
            radial_coordinate * jnp.sin(phi),
        ],
        axis=-1,
    )
    return plane, is_north


def spinor_coordinates_from_stereographic(
    plane: jnp.ndarray,
    is_north: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a regular local monopole spinor.

    Args:
        plane: Local stereographic coordinates with trailing ``(x, y)`` axis.
        is_north: Whether the north chart was selected per electron.

    Returns:
        The regular local monopole spinor ``(u, v)``.
    """
    x, y = plane[..., 0], plane[..., 1]
    radius_squared = x**2 + y**2
    denominator = 1 + radius_squared
    spinor_norm = jnp.sqrt(denominator)

    north_u = 1 / spinor_norm
    north_v = (x - 1j * y) / spinor_norm
    south_u = (x + 1j * y) / spinor_norm
    south_v = 1 / spinor_norm
    u = jnp.where(is_north, north_u, south_u)
    v = jnp.where(is_north, north_v, south_v)
    return u, v


def stereographic_monopole_connection(
    plane: jnp.ndarray, is_north: jnp.ndarray, monopole_strength: float
) -> jnp.ndarray:
    r"""Monopole connection :math:`A\propto(-y,x)` in the local chart.

    This is the regular local gauge matching the spinors returned by
    :func:`spinor_coordinates_from_stereographic`: it is the symmetric-gauge
    potential :math:`A^{\mathrm{sym}}_\phi = Q\cos\theta/\sin\theta` shifted by
    the gauge transformation :math:`\mp Q\phi` that removes the Dirac string
    from the chart's own pole.

    Args:
        plane: Local stereographic coordinates with trailing ``(x, y)`` axis.
        is_north: Whether the north chart was selected per electron.
        monopole_strength: Monopole strength :math:`Q = \mathrm{flux}/2`.

    Returns:
        Connection one-form with the same ``(..., 2)`` shape as ``plane``.
    """
    x, y = plane[..., 0], plane[..., 1]
    chart_sign = jnp.where(is_north, -1.0, 1.0)
    scale = 2 * chart_sign * monopole_strength / (1 + x**2 + y**2)
    return jnp.stack([-scale * y, scale * x], axis=-1)


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
