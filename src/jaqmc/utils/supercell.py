# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
    r"""Generates supercell k-points in a primitive reciprocal fundamental domain.

    These are the k-points of the primitive cell that fold into the Gamma point
    of the supercell. They satisfy the condition:

    .. math::
        \mathbf{k} \cdot \mathbf{S}^{-1} \pmod 1 = 0

    **Algorithm Explanation**:

    This function finds integer vectors :math:`\mathbf{n}` such that the fractional
    coordinates :math:`\mathbf{n} \cdot \mathbf{S}^{-T}` lie in the primitive
    reciprocal fundamental parallelepiped. This matches the row-vector convention
    used by ``supercell_lattice = S @ lattice``.

    For non-diagonal :math:`\mathbf{S}` (e.g., transforming an FCC primitive cell to a
    conventional cell), the valid integers :math:`\mathbf{n}` form a skewed volume.
    The algorithm:

    1. Finds the bounding box of this skewed volume in integer space.
    2. Scans all integers within the box.
    3. Filters for points that map back into the unit cube.

    Args:
        S: Supercell matrix with shape (ndim, ndim).
        original_reciprocal_vectors: Reciprocal vectors of the primitive cell
            with shape (ndim, ndim).

    Returns:
        Array of k-points with shape (N_k, ndim).
    """
    frac_kpts = supercell_fractional_kpts(jnp.asarray(S))
    frac_kpts = jnp.asarray(frac_kpts, dtype=original_reciprocal_vectors.dtype)
    return frac_kpts @ original_reciprocal_vectors


def fold_to_reciprocal_voronoi(
    kpts: jnp.ndarray, reciprocal: jnp.ndarray
) -> jnp.ndarray:
    """Folds k-points to nearest reciprocal-lattice images.

    Args:
        kpts: Cartesian k-points with shape ``(nk, 2)``.
        reciprocal: Reciprocal lattice vectors with shape ``(2, 2)``.

    Returns:
        Cartesian k-points shifted by reciprocal lattice vectors so that each
        point lies in the nearest image around the origin.
    """
    shifts_1d = jnp.asarray([-1, 0, 1], dtype=kpts.dtype)
    sx, sy = jnp.meshgrid(shifts_1d, shifts_1d, indexing="ij")
    shifts = jnp.stack([sx.reshape(-1), sy.reshape(-1)], axis=-1) @ reciprocal
    choices = kpts[:, None, :] + shifts[None, :, :]
    # Nudge the center toward the bottom-left so ties between equidistant images
    # resolve consistently (keep the top-right corner, drop its mirror).
    distances = jnp.linalg.norm(choices + jnp.asarray(-1e-10, kpts.dtype), axis=-1)
    idx = jnp.argmin(distances, axis=-1)
    return choices[jnp.arange(kpts.shape[0]), idx]


def get_supercell_kpts_in_first_bz(
    supercell_matrix: jnp.ndarray,
    primitive_lattice: jnp.ndarray,
) -> jnp.ndarray:
    r"""Computes primitive-cell momenta that fold to supercell Gamma.

    The returned set contains the primitive-cell momenta :math:`\mathbf k`
    satisfying

    .. math::

        S^T \mathbf k_{\rm frac} \in \mathbb Z^2,

    where :math:`S` is the integer supercell matrix and
    :math:`\mathbf k_{\rm frac}` is expressed in the primitive reciprocal
    basis. The Cartesian representatives are folded into the primitive
    reciprocal Voronoi cell, i.e. the primitive-cell first Brillouin zone.

    Args:
        supercell_matrix: Integer supercell matrix with shape ``(2, 2)``.
        primitive_lattice: Direct lattice vectors of the primitive cell.

    Returns:
        Primitive-cell Cartesian momenta that fold to supercell Gamma,
        represented in the primitive reciprocal Voronoi/first Brillouin zone.
    """
    primitive_lattice = jnp.asarray(primitive_lattice)
    reciprocal = get_reciprocal_vectors(primitive_lattice)
    raw_kpts = get_supercell_kpts(supercell_matrix, reciprocal)
    return fold_to_reciprocal_voronoi(raw_kpts, reciprocal)


def _fundamental_cell_indices(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return integer representatives in the fundamental cell of ``matrix``.

    The returned integer vectors ``n`` satisfy ``n @ inv(matrix)`` in
    ``[0, 1)`` along every axis. Membership is tested in integer arithmetic to
    avoid floating-point misclassification at cell boundaries. Writing
    ``adj = det(matrix) * inv(matrix)``, the equivalent integer condition is
    ``0 <= sign(det) * n @ adj < abs(det)``.

    Enumeration is eager and requires ``matrix`` to be a concrete integer
    array because its values determine the candidate-grid shape.
    """
    matrix_int = jnp.rint(jnp.asarray(matrix)).astype(int)
    matrix_inexact = matrix_int.astype(float)
    ndim = matrix_int.shape[-1]
    det = round(float(jnp.linalg.det(matrix_inexact)))
    adj = jnp.rint(jnp.linalg.inv(matrix_inexact) * det).astype(int)

    corners = jnp.stack(
        [x.ravel() for x in jnp.meshgrid(*([jnp.array([0, 1])] * ndim), indexing="ij")],
        axis=-1,
    )
    transformed = corners @ matrix_int
    n_min = jnp.amin(transformed, axis=0)
    n_max = jnp.amax(transformed, axis=0)
    possible_indices = jnp.stack(
        [
            x.ravel()
            for x in jnp.meshgrid(*list(map(jnp.arange, n_min, n_max)), indexing="ij")
        ],
        axis=-1,
    )

    scaled_fractional = (possible_indices @ adj) * jnp.sign(det)
    in_fundamental_cell = (scaled_fractional >= 0) & (scaled_fractional < abs(det))
    return possible_indices[jnp.all(in_fundamental_cell, axis=1)]


def supercell_fractional_kpts(supercell_matrix: jnp.ndarray) -> jnp.ndarray:
    r"""Returns primitive fractional k-points that fold to supercell Gamma.

    These are the primitive-cell momenta whose fractional coordinates
    :math:`\mathbf{n} \cdot \mathbf{S}^{-T}` lie within the primitive Brillouin
    zone, matching the row-vector convention ``supercell_lattice = S @ lattice``.
    The enumeration is performed eagerly because the supercell matrix must be a
    concrete integer matrix.

    Args:
        supercell_matrix: Integer supercell matrix :math:`S` with shape
            ``(ndim, ndim)``.

    Returns:
        Fractional primitive reciprocal coordinates with shape
        ``(abs(det(S)), ndim)``.
    """
    matrix = jnp.asarray(supercell_matrix).T
    indices = _fundamental_cell_indices(matrix)
    fractional_kpts = jnp.matmul(
        indices, jnp.linalg.inv(matrix.astype(float)), precision="highest"
    )
    return jnp.mod(fractional_kpts, 1.0)


def get_supercell_copies(latvec: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    r"""Calculates translation vectors to tile the supercell with the primitive cell.

    The vectors :math:`\mathbf{R}` are used to map the primitive cell to the supercell.

    Args:
        latvec: Primitive direct lattice vectors with shape ``(ndim, ndim)``.
        S: Integer supercell matrix with shape ``(ndim, ndim)``.

    Returns:
        Translation vectors with shape ``(abs(det(S)), ndim)``.
    """
    indices = _fundamental_cell_indices(S)
    return indices.astype(latvec.dtype) @ latvec
