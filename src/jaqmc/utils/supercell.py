# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from jax import numpy as jnp

LatticeType = Literal["cubic", "fcc", "bcc", "hexagonal", "honey"]


def get_reciprocal_vectors(lattice: jnp.ndarray) -> jnp.ndarray:
    r"""Computes reciprocal lattice vectors.

    Formula:
        :math:`\mathbf{b}_i = 2\pi (\mathbf{a}^{-1})^T`

    Args:
        lattice: The lattice vectors.

    Returns:
        Reciprocal lattice vectors.
    """
    return 2 * jnp.pi * jnp.linalg.inv(lattice).T


def get_supercell_kpts(
    S: jnp.ndarray,
    original_reciprocal_vectors: jnp.ndarray,
) -> jnp.ndarray:
    r"""Generates supercell k-points in the primitive cell's first Brillouin zone.

    These are the k-points of the primitive cell that fold into the Gamma point
    of the supercell. They satisfy the condition:

    .. math::
        \mathbf{k} \cdot \mathbf{S}^{-1} \pmod 1 = 0

    **Algorithm Explanation**:

    This function finds integer vectors :math:`\mathbf{n}` such that the fractional
    coordinates :math:`\mathbf{n} \cdot \mathbf{S}^{-1}` lie within the primitive
    Brillouin Zone.

    For non-diagonal :math:`\mathbf{S}` (e.g., transforming an FCC primitive cell to a
    conventional cell), the valid integers :math:`\mathbf{n}` form a skewed volume.
    The algorithm:

    1. Finds the bounding box of this skewed volume in integer space.
    2. Scans all integers within the box.
    3. Filters for points that map back into the unit cube.

    Args:
        S: Supercell matrix with shape (3, 3).
        original_reciprocal_vectors: Reciprocal vectors of the primitive cell (3, 3).

    Returns:
        Array of k-points with shape (N_k, 3).
    """
    ST = S.T
    u = jnp.array([0, 1])
    mesh_u = jnp.meshgrid(*[u] * 3, indexing="ij")
    unit_box_corners = jnp.stack([x.ravel() for x in mesh_u], axis=-1)

    # Transform unit box corners to find the bounding box of integers n
    transformed_corners = jnp.dot(unit_box_corners, ST)
    n_min = jnp.floor(jnp.amin(transformed_corners, axis=0)).astype(int)
    n_max = jnp.ceil(jnp.amax(transformed_corners, axis=0)).astype(int)

    # Scan the grid
    ranges = [jnp.arange(mi, ma + 1) for mi, ma in zip(n_min, n_max)]
    mesh = jnp.meshgrid(*ranges, indexing="ij")
    possible_n = jnp.stack([x.ravel() for x in mesh], axis=-1)

    # Map back to fractional coords: k_frac = n @ inv(S)
    possible_k_frac = jnp.dot(possible_n, jnp.linalg.inv(S))

    # Filter: keep points in [0, 1) with tolerance
    tol = 1e-5
    in_box = (possible_k_frac >= -tol) & (possible_k_frac < 1.0 - tol)
    mask = jnp.all(in_box, axis=1)

    valid_k_frac = possible_k_frac[mask]
    valid_k_frac = jnp.mod(valid_k_frac, 1.0)
    kpts = jnp.dot(valid_k_frac, original_reciprocal_vectors)
    return kpts


def get_supercell_copies(latvec: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    r"""Calculates translation vectors to tile the supercell with the primitive cell.

    The vectors :math:`\mathbf{R}` are used to map the primitive cell to the supercell.

    Args:
        latvec: Primitive lattice vectors (3, 3).
        S: Supercell matrix with shape (3, 3).

    Returns:
        Array of translation vectors with shape (N_cells, 3).
    """
    u = jnp.array([0, 1])
    mesh_u = jnp.meshgrid(*[u] * 3, indexing="ij")
    unit_box = jnp.stack([x.ravel() for x in mesh_u], axis=-1)
    unit_box_ = jnp.dot(unit_box, S)
    xyz_min = jnp.amin(unit_box_, axis=0)
    xyz_max = jnp.amax(unit_box_, axis=0)

    ranges = [
        jnp.arange(jnp.floor(mi), jnp.ceil(ma)) for mi, ma in zip(xyz_min, xyz_max)
    ]
    mesh = jnp.meshgrid(*ranges, indexing="ij")
    possible_pts_indices = jnp.stack([x.ravel() for x in mesh], axis=-1)

    possible_pts_frac = jnp.dot(possible_pts_indices, jnp.linalg.inv(S))

    in_unit_box = (possible_pts_frac >= -1e-5) & (possible_pts_frac < 1 - 1e-5)
    mask = jnp.all(in_unit_box, axis=1)

    valid_indices = possible_pts_indices[mask]

    return jnp.dot(valid_indices, latvec)
