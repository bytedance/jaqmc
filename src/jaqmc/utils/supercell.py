# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp


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


def get_primitive_kpts_for_supercell(
    supercell_matrix: jnp.ndarray,
    primitive_reciprocal_vectors: jnp.ndarray,
    twist: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Return primitive-cell k-points compatible with a supercell boundary condition.

    The returned Cartesian k-points are expressed in the primitive reciprocal
    basis and satisfy the boundary condition of
    ``supercell_lattice = supercell_matrix @ primitive_lattice``. With no
    ``twist``, they fold to the supercell Gamma point. A twist shifts the entire
    mesh by its fractional coordinates in the supercell reciprocal basis:

    .. math::

        \mathbf{k} = \mathbf{k}_\Gamma + \boldsymbol{\theta} \cdot \mathbf{B}_S

    Equivalently, the primitive reciprocal fractional coordinates are shifted
    by :math:`\boldsymbol{\theta} \cdot S^{-T}`.

    The enumeration finds integer vectors :math:`\mathbf{n}` for which
    :math:`\mathbf{n} \cdot \mathbf{S}^{-T}` lies in
    :math:`[0, 1)^{\mathrm{ndim}}`. This matches the row-vector convention
    used by ``supercell_lattice = S @ lattice``.

    For a non-diagonal :math:`\mathbf{S}`, such as one that transforms an FCC
    primitive cell to a conventional cell, the valid integer vectors occupy a
    skewed region. The algorithm:

    1. Finds the bounding box of this skewed volume in integer space.
    2. Scans all integers within the box.
    3. Filters for points that map back into the unit cube.

    Args:
        supercell_matrix: Integer supercell matrix :math:`S` with shape
            ``(ndim, ndim)``.
        primitive_reciprocal_vectors: Primitive-cell reciprocal vectors with
            shape ``(ndim, ndim)``.
        twist: Optional twist in fractional supercell reciprocal coordinates
            with shape ``(ndim,)``. ``None`` selects Gamma boundary conditions.

    Returns:
        Primitive-cell Cartesian k-points with shape ``(abs(det(S)), ndim)``.
    """
    primitive_reciprocal_vectors = jnp.asarray(primitive_reciprocal_vectors)
    fractional_kpts = supercell_fractional_kpts(jnp.asarray(supercell_matrix))
    fractional_kpts = fractional_kpts.astype(primitive_reciprocal_vectors.dtype)
    if twist is not None:
        twist = jnp.asarray(twist, dtype=primitive_reciprocal_vectors.dtype)
        fractional_kpts += jnp.linalg.solve(
            jnp.asarray(supercell_matrix, dtype=primitive_reciprocal_vectors.dtype),
            twist,
        )
    return fractional_kpts @ primitive_reciprocal_vectors


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
    raw_kpts = get_primitive_kpts_for_supercell(supercell_matrix, reciprocal)
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
