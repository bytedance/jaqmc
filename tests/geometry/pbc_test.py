# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from jaqmc.geometry import pbc


def test_distance_diagonal_lattice():
    """Test optimized distance function for diagonal lattices."""
    # Simple cubic lattice 10x10x10
    lattice = jnp.diag(jnp.array([10.0, 10.0, 10.0]))

    dist_fn = pbc.build_distance_fn(lattice)

    # Case 1: Intra-Cell Check
    r1 = jnp.array([[1.0, 2.0, 3.0]])
    r2 = jnp.array([[4.0, 2.0, 3.0]])
    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 3.0)
    assert jnp.allclose(disp[0, 0], jnp.array([-3.0, 0.0, 0.0]))

    # Case 2: Cross-Boundary Check
    # 1.0 and 9.0 along x-axis (size 10). Distance should be 2.
    r1 = jnp.array([[1.0, 5.0, 5.0]])
    r2 = jnp.array([[9.0, 5.0, 5.0]])
    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 2.0)


def test_distance_diagonal_lattice_non_cubic():
    """Test optimized distance function for non-cubic diagonal lattices."""
    # Orthorhombic lattice: 10 x 20 x 5
    lattice = jnp.diag(jnp.array([10.0, 20.0, 5.0]))
    dist_fn = pbc.build_distance_fn(lattice)

    # Along x (size 10): 1.0 and 9.0 -> dist 2.0 (wrap)
    r1 = jnp.array([[1.0, 0.0, 0.0]])
    r2 = jnp.array([[9.0, 0.0, 0.0]])
    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 2.0)
    assert jnp.allclose(disp[0, 0], jnp.array([2.0, 0.0, 0.0]))

    # Along y (size 20): 1.0 and 19.0 -> dist 2.0 (wrap)
    r1 = jnp.array([[0.0, 1.0, 0.0]])
    r2 = jnp.array([[0.0, 19.0, 0.0]])
    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 2.0)
    assert jnp.allclose(disp[0, 0], jnp.array([0.0, 2.0, 0.0]))

    # Along z (size 5): 1.0 and 4.0 -> dist 2.0 (wrap)
    r1 = jnp.array([[0.0, 0.0, 1.0]])
    r2 = jnp.array([[0.0, 0.0, 4.0]])
    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 2.0)
    assert jnp.allclose(disp[0, 0], jnp.array([0.0, 0.0, 2.0]))


def test_distance_orthogonal_non_diagonal_lattice():
    """Test optimized distance function for orthogonal but non-diagonal lattices."""
    # Permuted lattice vectors: [[0, 10, 0], [10, 0, 0], [0, 0, 10]]
    # This is orthogonal but not diagonal.
    lattice = jnp.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])

    dist_fn = pbc.build_distance_fn(lattice)

    # Point A: [1, 1, 1]
    # Point B: [1, 9, 1]
    # Along y-axis. The first lattice vector is [0, 10, 0] (y-axis).
    # Distance should be 2.0 (1-9 = -8 -> wraps to 2 via [0, 10, 0])

    r1 = jnp.array([[1.0, 1.0, 1.0]])
    r2 = jnp.array([[1.0, 9.0, 1.0]])

    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 2.0)
    assert jnp.allclose(disp[0, 0], jnp.array([0.0, 2.0, 0.0]))


def test_distance_general_lattice():
    """Test general MIC distance function for triclinic lattices."""
    # 2D Hexagonal lattice in 3D (z is orthogonal)

    val = 5.0 * jnp.sqrt(3.0)
    lattice = jnp.array([[10.0, 0.0, 0.0], [5.0, val, 0.0], [0.0, 0.0, 10.0]])

    dist_fn = pbc.build_distance_fn(lattice)

    # Case 1: Intra-Cell Check
    # Point at [1, 1, 0] and [2, 2, 0]. Dist sqrt(2).
    r1 = jnp.array([[1.0, 1.0, 0.0]])
    r2 = jnp.array([[2.0, 2.0, 0.0]])
    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, jnp.sqrt(2.0))
    assert jnp.allclose(disp[0, 0], jnp.array([-1.0, -1.0, 0.0]))

    # Case 2: Cross-Boundary Check
    # Dist should be 0.1.
    r1 = jnp.array([[0.0, 0.0, 0.0]])
    r2 = jnp.array([[5.0, val - 0.1, 0.0]])

    # r1 - r2 = [-5, -8.56, 0].
    # Nearest image of r2 to r1 is r2 - a2 = [0, -0.1, 0].
    # So r1 - (image) = [0, 0.1, 0].

    disp, dist = dist_fn(r1, r2)
    assert jnp.allclose(dist, 0.1, atol=1e-5)
    assert jnp.allclose(disp[0, 0], jnp.array([0.0, 0.1, 0.0]))
