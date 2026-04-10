# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for geometry modules using mathematical properties as oracles.

OBC tests: antisymmetry, diagonal zeros, norm consistency, shapes.
PBC tests: boundary values of scaled_f/g, symmetry, cross-validation
    of diagonal/orthogonal branches against the general branch.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.geometry import obc, pbc

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
    np.testing.assert_allclose(pbc.scaled_f(0.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(pbc.scaled_f(jnp.pi), jnp.pi * 0.75, atol=1e-6)
    np.testing.assert_allclose(pbc.scaled_f(-jnp.pi), jnp.pi * 0.75, atol=1e-6)


def test_scaled_f_is_even():
    """f(-w) = f(w)."""
    w = jnp.linspace(-jnp.pi, jnp.pi, 100)
    np.testing.assert_allclose(pbc.scaled_f(-w), pbc.scaled_f(w), atol=1e-7)


def test_scaled_g_boundary_values():
    """g(0) = 0 and g(+-pi) = 0."""
    np.testing.assert_allclose(pbc.scaled_g(0.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(pbc.scaled_g(jnp.pi), 0.0, atol=1e-6)
    np.testing.assert_allclose(pbc.scaled_g(-jnp.pi), 0.0, atol=1e-6)


def test_scaled_g_is_odd():
    """g(-w) = -g(w)."""
    w = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 100)
    np.testing.assert_allclose(pbc.scaled_g(-w), -pbc.scaled_g(w), atol=1e-7)


def test_scaled_g_derivative_at_boundary():
    """g'(pi) should be -0.5 (analytical from the polynomial formula)."""
    import jax

    g_grad = jax.grad(pbc.scaled_g)(jnp.pi)
    np.testing.assert_allclose(float(g_grad), -0.5, atol=1e-5)


# -- PBC: cross-validate optimized branches against general branch -----


def _general_mic_distance(lattice, r1, r2):
    """Brute-force MIC distance by searching all 27 images (NumPy).

    Returns:
        Tuple of (displacement, distance).
    """
    lattice = np.asarray(lattice)
    r1, r2 = np.asarray(r1), np.asarray(r2)
    shifts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                shifts.append(np.array([i, j, k]) @ lattice)

    best_disp = None
    best_dist = np.inf
    for s in shifts:
        diff = r1 - r2 + s
        d = np.linalg.norm(diff)
        if d < best_dist:
            best_dist = d
            best_disp = diff
    return best_disp, best_dist


def _compare_branch_against_general(lattice, pairs):
    """Helper: build distance_fn from lattice, compare against brute-force."""
    dist_fn = pbc.build_distance_fn(lattice)
    for r1, r2 in pairs:
        r1_arr = jnp.array([r1])
        r2_arr = jnp.array([r2])
        disp, dist = dist_fn(r1_arr, r2_arr)

        expected_disp, expected_dist = _general_mic_distance(lattice, r1, r2)
        np.testing.assert_allclose(float(dist[0, 0]), expected_dist, atol=1e-5)
        np.testing.assert_allclose(np.array(disp[0, 0]), expected_disp, atol=1e-5)


def test_diagonal_branch_vs_general():
    """Diagonal lattice optimized path matches brute-force."""
    lattice = jnp.diag(jnp.array([8.0, 6.0, 10.0]))
    pairs = [
        ([1.0, 1.0, 1.0], [7.0, 5.0, 9.0]),  # wraps in all 3 axes
        ([0.0, 0.0, 0.0], [4.0, 3.0, 5.0]),  # exactly at half-cell
        ([2.0, 2.0, 2.0], [2.5, 2.5, 2.5]),  # small separation
    ]
    _compare_branch_against_general(lattice, pairs)


def test_orthogonal_branch_vs_general():
    """Orthogonal (non-diagonal) lattice matches brute-force."""
    # Permuted axes: orthogonal but not diagonal
    lattice = jnp.array([[0.0, 8.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    pairs = [
        ([1.0, 1.0, 1.0], [5.0, 7.0, 9.0]),
        ([0.0, 0.0, 0.0], [3.0, 4.0, 5.0]),
    ]
    _compare_branch_against_general(lattice, pairs)


def test_general_branch_vs_brute_force():
    """Triclinic lattice general path matches brute-force."""
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
    _compare_branch_against_general(lattice, pairs)
