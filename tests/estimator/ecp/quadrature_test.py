# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for spherical quadrature rules.

Uses mathematical properties as test oracles:
- Points lie on the unit sphere
- Weights sum to 1 (before 4*pi scaling)
- Integration of constant function gives 4*pi
- Integration of Y_10 (cos theta) gives 0 by symmetry
- Rotation matrices are orthogonal with det = 1
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.estimator.ecp.quadrature import (
    Icosahedron,
    Octahedron,
    get_quadrature,
)

ALL_QUADRATURES = [
    ("octahedron", Octahedron, 6),
    ("octahedron", Octahedron, 18),
    ("octahedron", Octahedron, 26),
    ("octahedron", Octahedron, 50),
    ("icosahedron", Icosahedron, 12),
    ("icosahedron", Icosahedron, 32),
]


@pytest.fixture(
    params=[
        pytest.param((cls, n), id=f"{name}_{n}") for name, cls, n in ALL_QUADRATURES
    ]
)
def quadrature(request):
    cls, n = request.param
    return cls(n)


def test_points_on_unit_sphere(quadrature):
    """All quadrature points should lie on the unit sphere."""
    norms = jnp.linalg.norm(quadrature.pts, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_correct_point_count(quadrature):
    """Point array shape should match declared n_points."""
    assert quadrature.pts.shape == (quadrature.n_points, 3)
    assert quadrature.coefs.shape == (quadrature.n_points,)


def test_weights_sum_to_one(quadrature):
    """Weights should sum to 1 (integrate() multiplies by 4pi separately)."""
    np.testing.assert_allclose(jnp.sum(quadrature.coefs), 1.0, atol=1e-6)


def test_integrate_constant(quadrature):
    """Integral of f=1 over the sphere should be 4*pi."""
    values = jnp.ones(quadrature.n_points)
    result = quadrature.integrate(values)
    np.testing.assert_allclose(result, 4 * jnp.pi, atol=1e-4)


def test_integrate_cos_theta_vanishes(quadrature):
    """Integral of cos(theta) = z-coordinate over the sphere should be 0."""
    cos_theta = quadrature.pts[:, 2]
    result = quadrature.integrate(cos_theta)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_integrate_x_vanishes(quadrature):
    """Integral of x over the sphere should be 0 by symmetry."""
    x = quadrature.pts[:, 0]
    result = quadrature.integrate(x)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_rotation_matrices_orthogonal():
    """Random rotation matrices should satisfy R @ R.T = I and det(R) = 1."""
    key = jax.random.key(42)
    n = 20
    matrices = Octahedron.sample_rotation_matrices(n, key)
    assert matrices.shape == (n, 3, 3)

    for i in range(n):
        r = matrices[i]
        np.testing.assert_allclose(r @ r.T, jnp.eye(3), atol=1e-5)
        np.testing.assert_allclose(jnp.linalg.det(r), 1.0, atol=1e-5)


def test_rotation_zero_samples():
    """n_samples=0 should return identity rotation."""
    key = jax.random.key(0)
    matrices = Octahedron.sample_rotation_matrices(0, key)
    assert matrices.shape == (1, 3, 3)
    np.testing.assert_allclose(matrices[0], jnp.eye(3), atol=1e-12)


def test_rotated_points_stay_on_sphere():
    """Rotated quadrature points should still lie on the unit sphere."""
    quad = Icosahedron(12)
    key = jax.random.key(7)
    rotated = quad.sample_rotated_points(5, key)
    assert rotated.shape == (5, 12, 3)

    norms = jnp.linalg.norm(rotated, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_get_quadrature_returns_correct_type():
    """get_quadrature should return the right type and cache instances."""
    q1 = get_quadrature("icosahedron_12")
    assert isinstance(q1, Icosahedron)
    assert q1.n_points == 12

    q2 = get_quadrature("octahedron_6")
    assert isinstance(q2, Octahedron)
    assert q2.n_points == 6

    # Same id returns same instance (caching)
    q3 = get_quadrature("icosahedron_12")
    assert q3 is q1


def test_get_quadrature_invalid():
    """Invalid quadrature ids should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid quadrature_id format"):
        get_quadrature("bad")
    with pytest.raises(ValueError, match="Unknown quadrature type"):
        get_quadrature("tetrahedron_4")


def test_invalid_point_counts():
    """Unsupported point counts should raise ValueError."""
    with pytest.raises(ValueError):
        Octahedron(7)
    with pytest.raises(ValueError):
        Icosahedron(20)
