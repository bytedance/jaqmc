# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for spherical quadrature rules.

Uses mathematical properties as test oracles:
- Points lie on the unit sphere
- Weights sum to 1 (before 4*pi scaling)
- Integration of constant function gives 4*pi
- Integration of Y_10 (cos theta) gives 0 by symmetry
- Rotation matrices are orthogonal with det = 1
"""

import math

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.estimator.ecp.quadrature import (
    ECPQuadrature,
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

EXACT_DEGREES = {
    ("octahedron", 6): 3,
    ("octahedron", 18): 5,
    ("octahedron", 26): 7,
    ("octahedron", 50): 11,
    ("icosahedron", 12): 5,
    ("icosahedron", 32): 9,
}


def _double_factorial(n: int) -> int:
    return math.prod(range(n, 0, -2))


def _sphere_average_monomial(a: int, b: int, c: int) -> float:
    if any(power % 2 for power in (a, b, c)):
        return 0.0
    numerator = math.prod(_double_factorial(power - 1) for power in (a, b, c))
    return numerator / _double_factorial(a + b + c + 1)


def _legacy_two_angle_rotations(n_samples: int, key) -> jnp.ndarray:
    """Old polar-axis-only construction used as a discriminating control."""
    phi_key, theta_key = jax.random.split(key)
    phi = jax.random.uniform(phi_key, shape=(n_samples,)) * 2 * jnp.pi
    cos_theta = 1 - 2 * jax.random.uniform(theta_key, shape=(n_samples,))
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    entries = (
        sin_phi**2 + cos_theta * cos_phi**2,
        sin_phi * cos_phi * (cos_theta - 1),
        sin_theta * cos_phi,
        sin_phi * cos_phi * (cos_theta - 1),
        cos_phi**2 + cos_theta * sin_phi**2,
        sin_theta * sin_phi,
        -sin_theta * cos_phi,
        -sin_theta * sin_phi,
        cos_theta,
    )
    return jnp.stack(entries, axis=-1).reshape(n_samples, 3, 3)


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


@pytest.mark.parametrize(
    ("name", "quadrature_cls", "n_points"),
    ALL_QUADRATURES,
    ids=str,
)
def test_polynomial_exactness(name, quadrature_cls, n_points):
    """Every rule integrates all Cartesian monomials through its design order."""
    quadrature = quadrature_cls(n_points)
    points = np.asarray(quadrature.pts, dtype=np.float64)
    weights = np.asarray(quadrature.coefs, dtype=np.float64)
    exact_degree = EXACT_DEGREES[name, n_points]

    for total_degree in range(exact_degree + 1):
        for a in range(total_degree + 1):
            for b in range(total_degree - a + 1):
                c = total_degree - a - b
                actual = np.sum(
                    weights * points[:, 0] ** a * points[:, 1] ** b * points[:, 2] ** c
                )
                expected = _sphere_average_monomial(a, b, c)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=8e-7,
                    rtol=0.0,
                    err_msg=(
                        f"{name}_{n_points} failed x^{a} y^{b} z^{c} "
                        f"at total degree {total_degree}"
                    ),
                )


def test_rotation_matrices_orthogonal():
    """Random rotation matrices should satisfy R @ R.T = I and det(R) = 1."""
    key = jax.random.key(42)
    n = 20
    matrices = Octahedron.sample_rotation_matrices(n, key)
    assert matrices.shape == (n, 3, 3)

    np.testing.assert_allclose(
        matrices @ jnp.swapaxes(matrices, -1, -2),
        jnp.broadcast_to(jnp.eye(3), matrices.shape),
        atol=1e-5,
    )
    np.testing.assert_allclose(jnp.linalg.det(matrices), 1.0, atol=1e-5)


def test_rotation_sampler_is_haar_and_control_is_not():
    """Haar moments pass while the former two-angle construction clearly fails."""
    key = jax.random.key(43)
    matrices = Octahedron.sample_rotation_matrices(16_384, key)
    legacy = _legacy_two_angle_rotations(16_384, key)

    np.testing.assert_allclose(jnp.mean(matrices, axis=0), 0.0, atol=2e-2)
    np.testing.assert_allclose(
        jnp.mean(jnp.square(matrices), axis=0),
        jnp.full((3, 3), 1.0 / 3.0),
        atol=2e-2,
    )
    assert float(jnp.mean(legacy[:, 0, 0])) > 0.45
    assert float(jnp.mean(jnp.square(legacy[:, 0, 0]))) > 0.45


def test_rotation_zero_samples():
    """n_samples=0 should return identity rotation."""
    key = jax.random.key(0)
    matrices = Octahedron.sample_rotation_matrices(0, key)
    assert matrices.shape == (1, 3, 3)
    np.testing.assert_allclose(matrices[0], jnp.eye(3), atol=1e-12)


@pytest.mark.requires_x64
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_rotation_preserves_requested_dtype(dtype):
    matrices = Octahedron.sample_rotation_matrices(4, jax.random.key(9), dtype=dtype)
    assert matrices.dtype == dtype


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


def test_quadrature_options_are_discoverable():
    assert [choice.value for choice in ECPQuadrature] == [
        "octahedron_6",
        "octahedron_18",
        "octahedron_26",
        "octahedron_50",
        "icosahedron_12",
        "icosahedron_32",
    ]


def test_get_quadrature_invalid():
    """Invalid quadrature ids should raise ValueError."""
    with pytest.raises(ValueError, match="is not a valid ECPQuadrature"):
        get_quadrature("bad")
    with pytest.raises(ValueError, match="is not a valid ECPQuadrature"):
        get_quadrature("tetrahedron_4")


def test_invalid_point_counts():
    """Unsupported point counts should raise ValueError."""
    with pytest.raises(ValueError):
        Octahedron(7)
    with pytest.raises(ValueError):
        Icosahedron(20)
