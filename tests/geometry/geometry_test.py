# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for geometry modules using mathematical properties as oracles.

OBC tests: antisymmetry, diagonal zeros, norm consistency, shapes.
PBC tests: boundary values of scaled_f/g, symmetry, cross-validation
    of build_distance_fn against an independent exact minimum-image oracle.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.geometry import obc, pbc, sphere

# -- OBC: pair_displacements_within ------------------------------------


def test_within_antisymmetry():
    """disp[i,j] == -disp[j,i]."""
    pos = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]])
    disp, _ = obc.pair_displacements_within(pos)
    np.testing.assert_allclose(disp, -jnp.transpose(disp, (1, 0, 2)), atol=1e-7)


def test_within_diagonal_zeros():
    """Self-distances should be zero."""
    pos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    disp, r = obc.pair_displacements_within(pos)
    np.testing.assert_allclose(jnp.diagonal(r), 0.0, atol=1e-7)
    np.testing.assert_allclose(jnp.diagonal(disp, axis1=0, axis2=1).T, 0.0, atol=1e-7)


def test_within_norm_consistency():
    """||disp[i,j]|| should equal r[i,j] for off-diagonal entries."""
    pos = jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    disp, r = obc.pair_displacements_within(pos)
    n = pos.shape[0]
    mask = 1.0 - jnp.eye(n)
    computed_r = jnp.linalg.norm(disp, axis=-1) * mask
    np.testing.assert_allclose(computed_r, r, atol=1e-6)


def test_within_known_distance():
    """Check a specific known distance: (3,4,0) triangle → distance 5."""
    pos = jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    _, r = obc.pair_displacements_within(pos)
    np.testing.assert_allclose(r[0, 1], 5.0, atol=1e-6)
    np.testing.assert_allclose(r[1, 0], 5.0, atol=1e-6)


def test_within_rejects_bad_shape():
    """Should reject 1D input."""
    with pytest.raises(ValueError, match="n_particles, ndim"):
        obc.pair_displacements_within(jnp.array([1.0, 2.0, 3.0]))


# -- OBC: pair_displacements_between -----------------------------------


def test_between_shape():
    """Output shapes should be (n_a, n_b, ndim) and (n_a, n_b)."""
    a = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    b = jnp.array([[0.5, 0.5], [1.5, 1.5]])
    disp, r = obc.pair_displacements_between(a, b)
    assert disp.shape == (3, 2, 2)
    assert r.shape == (3, 2)


def test_between_norm_consistency():
    """||disp[i,j]|| should equal r[i,j] for all entries."""
    a = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = jnp.array([[3.0, 4.0, 0.0]])
    disp, r = obc.pair_displacements_between(a, b)
    np.testing.assert_allclose(jnp.linalg.norm(disp, axis=-1), r, atol=1e-6)


def test_between_known_distance():
    """Electron at origin, nucleus at (3,4,0) → distance 5."""
    electrons = jnp.array([[0.0, 0.0, 0.0]])
    nuclei = jnp.array([[3.0, 4.0, 0.0]])
    disp, r = obc.pair_displacements_between(electrons, nuclei)
    np.testing.assert_allclose(r[0, 0], 5.0, atol=1e-6)
    np.testing.assert_allclose(disp[0, 0], jnp.array([-3.0, -4.0, 0.0]), atol=1e-6)


def test_between_rejects_bad_input():
    """Should reject mismatched ndim or non-2D input."""
    with pytest.raises(ValueError, match="n_a, ndim"):
        obc.pair_displacements_between(jnp.array([1.0, 2.0]), jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="spatial dimensions"):
        obc.pair_displacements_between(jnp.ones((2, 3)), jnp.ones((2, 2)))


# -- PBC: scaled_f and scaled_g boundary conditions --------------------


def test_scaled_f_boundary_values():
    """f(0) = 0 and f(+-pi) = 0.75*pi."""
    np.testing.assert_allclose(pbc.scaled_f(jnp.array(0.0)), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        pbc.scaled_f(jnp.array(jnp.pi)), jnp.pi * 0.75, atol=1e-6
    )
    np.testing.assert_allclose(
        pbc.scaled_f(jnp.array(-jnp.pi)), jnp.pi * 0.75, atol=1e-6
    )


def test_scaled_f_is_even():
    """f(-w) = f(w)."""
    w = jnp.linspace(-jnp.pi, jnp.pi, 100)
    np.testing.assert_allclose(pbc.scaled_f(-w), pbc.scaled_f(w), atol=1e-7)


def test_scaled_g_boundary_values():
    """g(0) = 0 and g(+-pi) = 0."""
    np.testing.assert_allclose(pbc.scaled_g(jnp.array(0.0)), 0.0, atol=1e-12)
    np.testing.assert_allclose(pbc.scaled_g(jnp.array(jnp.pi)), 0.0, atol=1e-6)
    np.testing.assert_allclose(pbc.scaled_g(jnp.array(-jnp.pi)), 0.0, atol=1e-6)


def test_scaled_g_is_odd():
    """g(-w) = -g(w)."""
    w = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 100)
    np.testing.assert_allclose(pbc.scaled_g(-w), -pbc.scaled_g(w), atol=1e-7)


def test_scaled_g_derivative_at_boundary():
    """g'(pi) should be -0.5 (analytical from the polynomial formula)."""
    import jax

    g_grad = jax.grad(pbc.scaled_g)(jnp.pi)
    np.testing.assert_allclose(float(g_grad), -0.5, atol=1e-5)


# -- PBC: cross-validate build_distance_fn against exact oracle --------


def _true_mic_distance(lattice, r1, r2):
    """Exact minimum-image distance by bounded lattice enumeration (NumPy).

    This is an independent oracle: unlike the code under test, it does not
    assume that searching the 27 neighbouring cells is enough.

    The search box is sized from a triangle-inequality bound:

    1. Reduce the separation into one cell by subtracting a lattice vector.
       The reduced vector ``r0`` is itself a candidate image, and its length
       is bounded by the cell, not by how far apart the two points are.
    2. Since ``r0`` is a candidate, the best image cannot be longer than
       ``2 * |r0|``. The reciprocal basis converts that length bound into a
       per-axis integer bound.
    3. Enumerate that box. It is guaranteed to contain the global minimum.

    Returns:
        Tuple of (displacement, distance) for the global minimum image.
    """
    lattice = np.asarray(lattice, dtype=float)
    diff = np.asarray(r1, dtype=float) - np.asarray(r2, dtype=float)
    inv_lattice = np.linalg.inv(lattice)
    reduced = diff - np.round(diff @ inv_lattice) @ lattice
    # For a lattice vector v = m @ lattice, the i-th integer coefficient obeys
    # |m_i| = |b_i . v| / (2 pi) <= |b_i| |v| / (2 pi). With |v| <= 2 |reduced|
    # this bounds every coefficient, so the box holds the global minimum.
    reciprocal = 2.0 * np.pi * inv_lattice.T
    max_shift = np.ceil(
        np.linalg.norm(reciprocal, axis=1) * np.linalg.norm(reduced) / np.pi
    ).astype(int)
    ranges = [np.arange(-m, m + 1) for m in max_shift]
    shifts = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(
        -1, lattice.shape[0]
    )
    images = reduced + shifts @ lattice
    dists = np.linalg.norm(images, axis=-1)
    best = int(np.argmin(dists))
    return images[best], float(dists[best])


def _assert_distance_fn_matches_oracle(lattice, pairs):
    """Assert ``build_distance_fn`` matches the oracle on every pair."""
    dist_fn = pbc.build_distance_fn(lattice)
    inv_lattice = np.linalg.inv(np.asarray(lattice, dtype=float))
    for r1, r2 in pairs:
        disp, dist = dist_fn(jnp.array([r1]), jnp.array([r2]))
        _, expected_dist = _true_mic_distance(lattice, r1, r2)

        np.testing.assert_allclose(float(dist[0, 0]), expected_dist, atol=1e-5)
        np.testing.assert_allclose(
            float(jnp.linalg.norm(disp[0, 0])), expected_dist, atol=1e-5
        )
        # The returned displacement must differ from the raw separation by an
        # exact lattice vector (ties may pick a different but equal-length image).
        shift = np.asarray(disp[0, 0]) - (np.asarray(r1) - np.asarray(r2))
        frac_shift = shift @ inv_lattice
        np.testing.assert_allclose(frac_shift, np.round(frac_shift), atol=1e-6)


def test_diagonal_branch():
    """Diagonal lattice optimized path matches the exact oracle."""
    lattice = jnp.diag(jnp.array([8.0, 6.0, 10.0]))
    pairs = [
        ([1.0, 1.0, 1.0], [7.0, 5.0, 9.0]),  # wraps in all 3 axes
        ([0.0, 0.0, 0.0], [4.0, 3.0, 5.0]),  # exactly at half-cell
        ([2.0, 2.0, 2.0], [2.5, 2.5, 2.5]),  # small separation
    ]
    _assert_distance_fn_matches_oracle(lattice, pairs)


def test_orthogonal_branch():
    """Orthogonal (non-diagonal) lattice matches the exact oracle."""
    # Permuted axes: orthogonal but not diagonal
    lattice = jnp.array([[0.0, 8.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    pairs = [
        ([1.0, 1.0, 1.0], [5.0, 7.0, 9.0]),
        ([0.0, 0.0, 0.0], [3.0, 4.0, 5.0]),
    ]
    _assert_distance_fn_matches_oracle(lattice, pairs)


def test_general_branch():
    """Triclinic lattice general path matches the exact oracle."""
    lattice = jnp.array(
        [
            [8.0, 0.0, 0.0],
            [2.0, 7.0, 0.0],
            [1.0, 1.0, 9.0],
        ]
    )
    pairs = [
        ([0.5, 0.5, 0.5], [7.5, 6.5, 8.5]),  # near boundary
        ([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),  # interior
        ([0.0, 0.0, 0.0], [4.0, 3.5, 4.5]),  # from origin
    ]
    _assert_distance_fn_matches_oracle(lattice, pairs)


@pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(
            [[1.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], id="shear-xy"
        ),
        pytest.param(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], id="shear-xz"
        ),
        pytest.param(
            [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0], [3.0, 0.0, 1.0]], id="combined"
        ),
    ],
)
def test_general_branch_skewed(lattice):
    """Skewed triclinic cells need images far beyond the 27 neighbours."""
    # Separations whose minimum image sits many cells away in the original basis.
    pairs = [
        ([-6.0, -6.0, 0.5], [0.0, 0.0, 0.0]),
        ([-6.0, -5.0, 0.5], [0.0, 0.0, 0.0]),
        ([-4.0, -3.0, 0.5], [0.0, 0.0, 0.0]),
    ]
    _assert_distance_fn_matches_oracle(lattice, pairs)


# -- Sphere: projective spinor conversion ------------------------------


def test_cartesian_from_spinor_is_scale_invariant():
    u = jnp.array([1.0 + 2.0j, -0.5 + 0.25j])
    v = jnp.array([0.3 - 0.7j, 2.0 + 1.0j])
    scale = jnp.array([2.0 - 3.0j, -0.2 + 0.4j])

    expected = sphere.cartesian_from_spinor(u, v)
    actual = sphere.cartesian_from_spinor(scale * u, scale * v)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(actual, axis=-1), 1, atol=1e-6)


def test_cartesian_from_spinor_has_correct_second_derivative_at_pole():
    def z_coordinate(v_real):
        return sphere.cartesian_from_spinor(jnp.asarray(1.0 + 0.0j), v_real + 0.0j)[2]

    second_derivative = jax.grad(jax.grad(z_coordinate))(jnp.asarray(0.0))

    np.testing.assert_allclose(second_derivative, -4.0, atol=1e-6)
