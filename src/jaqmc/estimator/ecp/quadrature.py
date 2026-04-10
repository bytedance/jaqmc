# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Spherical quadrature rules for ECP nonlocal integration.

This module provides numerical integration over the unit sphere using
polyhedron-based quadrature rules (icosahedral and octahedral).
"""

from typing import ClassVar

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey

__all__ = ["Icosahedron", "Octahedron", "Quadrature", "get_quadrature"]

DEFAULT_QUADRATURE_ID = "icosahedron_12"


def _expand_sign(values: list[float]) -> list[list[float]]:
    """Expand a point by all sign permutations.

    Args:
        values: List of coordinate values to expand.

    Returns:
        List of all sign permutations, excluding zero values from sign flipping.
        Example: [a, b, c] => [[+a, +b, +c], [+a, +b, -c], ..., [-a, -b, -c]]
    """
    if len(values) == 1:
        result = [[values[0]]]
        if values[0] != 0:
            result.append([-values[0]])
        return result

    rest = _expand_sign(values[1:])
    result = [[values[0], *r] for r in rest]
    if values[0] != 0:
        result.extend([[-values[0], *r] for r in rest])
    return result


class Quadrature:
    """Base class for spherical quadrature rules.

    Provides numerical integration over the unit sphere using a set of
    quadrature points and weights. Random rotations are used to remove
    bias from the fixed quadrature grid.

    Args:
        n_points: Number of quadrature points.
    """

    pts: jnp.ndarray  # Shape: (n_points, 3)
    coefs: jnp.ndarray  # Shape: (n_points,)

    def __init__(self, n_points: int) -> None:
        self.n_points = n_points

    @staticmethod
    def sample_rotation_matrices(n_samples: int, key: PRNGKey) -> jnp.ndarray:
        """Sample random rotation matrices for unbiased integration.

        Args:
            n_samples: Number of rotation matrices to sample.
            key: PRNG key for random rotation.

        Returns:
            Rotation matrices of shape (n_samples, 3, 3).
        """
        if n_samples == 0:
            return jnp.eye(3)[None, ...]

        key, subkey = jax.random.split(key)
        phi = jax.random.uniform(subkey, shape=(n_samples,)) * jnp.pi * 2

        key, subkey = jax.random.split(key)
        cos_theta = 1.0 - 2 * jax.random.uniform(subkey, shape=(n_samples,))
        sin_theta = jnp.sqrt(1.0 - cos_theta**2)

        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)
        sin_phi2 = sin_phi**2
        cos_phi2 = cos_phi**2

        # Build rotation matrix components
        m11 = sin_phi2 + cos_theta * cos_phi2
        m12 = sin_phi * cos_phi * (cos_theta - 1)
        m13 = sin_theta * cos_phi

        m21 = m12
        m22 = cos_phi2 + cos_theta * sin_phi2
        m23 = sin_theta * sin_phi

        m31 = -m13
        m32 = -m23
        m33 = cos_theta

        matrices = jnp.stack(
            [m11, m12, m13, m21, m22, m23, m31, m32, m33], axis=-1
        ).reshape(-1, 3, 3)
        return matrices

    def sample_rotated_points(self, n_samples: int, key: PRNGKey) -> jnp.ndarray:
        """Generate randomly rotated quadrature points for unbiased integration.

        Args:
            n_samples: Number of independent rotations (typically n_electrons).
            key: PRNG key for random rotation.

        Returns:
            Rotated points of shape (n_samples, n_points, 3).
        """
        rotation_matrices = self.sample_rotation_matrices(n_samples, key)
        # Apply rotation via einsum
        rotated = jnp.einsum("ijk,lk->ilj", rotation_matrices, self.pts)
        return rotated

    def integrate(self, values: jnp.ndarray) -> jnp.ndarray:
        r"""Weighted sum over quadrature points with :math:`4\pi` normalization.

        Args:
            values: Function values at quadrature points.
                Last axis should be n_points.

        Returns:
            Integral approximation (scaled by :math:`4\pi`).
        """
        return jnp.sum(values * self.coefs, axis=-1) * 4 * jnp.pi


class Octahedron(Quadrature):
    """Octahedral quadrature with 6, 18, 26, or 50 points.

    Uses vertices and edge/face centers of an octahedron with appropriate
    weights for polynomial exactness.
    """

    _coef_table: ClassVar[dict[int, list[float]]] = {
        6: [1.0 / 6.0] * 6,
        18: [1.0 / 30.0] * 6 + [1.0 / 15.0] * 12,
        26: [1.0 / 21.0] * 6 + [4.0 / 105.0] * 12 + [27.0 / 840.0] * 8,
        50: (
            [4.0 / 315.0] * 6
            + [64.0 / 2835.0] * 12
            + [27.0 / 1280.0] * 8
            + [14641.0 / 725760.0] * 24
        ),
    }

    def __init__(self, n_points: int) -> None:
        super().__init__(n_points)
        if n_points not in self._coef_table:
            raise ValueError(
                f"Octahedron quadrature supports 6, 18, 26, or 50 points, "
                f"got {n_points}"
            )

        # Build points: vertices (A), edge midpoints (B), face centers (C),
        # and additional points (D)
        pts_list: list[list[float]] = []

        # A: 6 vertices along axes
        pts_list.extend(_expand_sign([1.0, 0.0, 0.0]))
        pts_list.extend(_expand_sign([0.0, 1.0, 0.0]))
        pts_list.extend(_expand_sign([0.0, 0.0, 1.0]))

        # B: 12 edge midpoints
        p = 1.0 / jnp.sqrt(2.0)
        pts_list.extend(_expand_sign([float(p), float(p), 0.0]))
        pts_list.extend(_expand_sign([float(p), 0.0, float(p)]))
        pts_list.extend(_expand_sign([0.0, float(p), float(p)]))

        # C: 8 face centers
        q = 1.0 / jnp.sqrt(3.0)
        pts_list.extend(_expand_sign([float(q), float(q), float(q)]))

        # D: 24 additional points for 50-point rule
        r = 1.0 / jnp.sqrt(11.0)
        s = 3.0 / jnp.sqrt(11.0)
        pts_list.extend(_expand_sign([float(r), float(r), float(s)]))
        pts_list.extend(_expand_sign([float(r), float(s), float(r)]))
        pts_list.extend(_expand_sign([float(s), float(r), float(r)]))

        self.pts = jnp.array(pts_list[:n_points])
        self.coefs = jnp.array(self._coef_table[n_points])


class Icosahedron(Quadrature):
    """Icosahedral quadrature with 12 or 32 points.

    Uses vertices of an icosahedron (12 points) or icosahedron plus
    dodecahedron face centers (32 points).
    """

    _coef_table: ClassVar[dict[int, list[float]]] = {
        12: [1.0 / 12.0] * 12,
        32: [5.0 / 168.0] * 12 + [27.0 / 840.0] * 20,
    }

    def __init__(self, n_points: int) -> None:
        super().__init__(n_points)
        if n_points not in self._coef_table:
            raise ValueError(
                f"Icosahedron quadrature supports 12 or 32 points, got {n_points}"
            )

        # Build points in spherical coordinates (theta, phi)
        polars: list[tuple[float, float]] = [
            (0.0, 0.0),
            (float(jnp.pi), 0.0),
        ]

        # B: 10 vertices of icosahedron (5 upper ring, 5 lower ring)
        arctan2 = float(jnp.arctan(2.0))
        for k in range(5):
            polars.append((arctan2, 2 * k * jnp.pi / 5))
        for k in range(5):
            polars.append((float(jnp.pi) - arctan2, (2 * k + 1) * jnp.pi / 5))

        # C: 20 face centers for 32-point rule
        down = float(jnp.sqrt(15 + 6 * jnp.sqrt(5.0)))
        theta1 = float(jnp.arccos((2 + jnp.sqrt(5.0)) / down))
        theta2 = float(jnp.arccos(1.0 / down))

        for k in range(5):
            polars.append((theta1, (2 * k + 1) * jnp.pi / 5))
        for k in range(5):
            polars.append((theta2, (2 * k + 1) * jnp.pi / 5))
        for k in range(5):
            polars.append((float(jnp.pi) - theta1, 2 * k * jnp.pi / 5))
        for k in range(5):
            polars.append((float(jnp.pi) - theta2, 2 * k * jnp.pi / 5))

        # Convert to Cartesian coordinates
        pts_list = [
            [
                float(jnp.sin(theta) * jnp.cos(phi)),
                float(jnp.sin(theta) * jnp.sin(phi)),
                float(jnp.cos(theta)),
            ]
            for theta, phi in polars[:n_points]
        ]
        self.pts = jnp.array(pts_list)
        self.coefs = jnp.array(self._coef_table[n_points])


_QUADRATURE_REGISTRY: dict[str, Quadrature] = {}


def get_quadrature(quadrature_id: str | None = None) -> Quadrature:
    """Get a quadrature instance by identifier.

    Args:
        quadrature_id: Quadrature identifier in format "{type}_{n_points}".
            Supported: "octahedron_6", "octahedron_26", "icosahedron_12".
            If None, uses the default "icosahedron_12".

    Returns:
        Quadrature instance.

    Raises:
        ValueError: If quadrature_id format is invalid or type is unknown.
    """
    quadrature_id = quadrature_id or DEFAULT_QUADRATURE_ID

    if quadrature_id not in _QUADRATURE_REGISTRY:
        parts = quadrature_id.split("_")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid quadrature_id format: {quadrature_id}. "
                "Expected format: 'type_npoints' (e.g., 'icosahedron_12')"
            )
        quad_type, n_points_str = parts
        n_points = int(n_points_str)

        if quad_type == "octahedron":
            _QUADRATURE_REGISTRY[quadrature_id] = Octahedron(n_points)
        elif quad_type == "icosahedron":
            _QUADRATURE_REGISTRY[quadrature_id] = Icosahedron(n_points)
        else:
            raise ValueError(
                f"Unknown quadrature type: {quad_type}. "
                "Supported: 'octahedron', 'icosahedron'"
            )

    return _QUADRATURE_REGISTRY[quadrature_id]
