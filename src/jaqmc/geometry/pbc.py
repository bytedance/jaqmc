# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.sampler.mcmc import SamplingProposal, gaussian_proposal


class DistanceType(StrEnum):
    r"""Periodic distance functions for solid-state systems.

    Maps particle separations into smooth representations that respect
    periodicity. Both functions take an (over-complete) set of real-space
    lattice vectors :math:`\mathbf{a}_i` and reciprocal vectors
    :math:`\mathbf{b}_i` produced by :func:`get_symmetry_lat`, and compute
    fractional projections :math:`\omega_i = \mathbf{b}_i \cdot \mathbf{r}`
    (wrapped to :math:`[-\pi, \pi]`).

    Attributes:
        nu: Polynomial distance. Defines two smooth, periodic-compatible
            polynomials:

            .. math::
                f(\omega) = \lvert\omega\rvert
                    \bigl(1 - \tfrac{1}{4}\lvert\omega/\pi\rvert^3\bigr)

            .. math::
                g(\omega) = \omega
                    \bigl(1 - \tfrac{3}{2}\lvert\omega/\pi\rvert
                    + \tfrac{1}{2}\lvert\omega/\pi\rvert^2\bigr)

            and computes the distance as:

            .. math::
                d(\mathbf{r}) = \sqrt{
                    \sum_i \lVert\mathbf{a}_i\rVert^2 f(\omega_i)^2
                    + \sum_{i \neq j} (\mathbf{a}_i \cdot \mathbf{a}_j)
                    \, g(\omega_i)\, g(\omega_j)
                }

            Produces 3D relative coordinates
            :math:`\sum_i g(\omega_i)\,\mathbf{a}_i`.
            Works well for most systems.
        tri: Trigonometric distance. Uses the metric tensor
            :math:`G_{ij} = \mathbf{a}_i \cdot \mathbf{a}_j` and:

            .. math::
                V_{ij} = \sin(\omega_i)\sin(\omega_j)
                    + (1 - \cos(\omega_i))(1 - \cos(\omega_j))

            to compute:

            .. math::
                d(\mathbf{r}) = \sqrt{\sum_{ij} V_{ij}\, G_{ij}}

            Produces 6D relative coordinates by concatenating
            :math:`\sum_i \sin(\omega_i)\,\mathbf{a}_i` and
            :math:`\sum_i \cos(\omega_i)\,\mathbf{a}_i`.
            More expressive than ``nu`` at the cost of doubling the feature
            dimension.
    """

    nu = "nu"
    tri = "tri"


class SymmetryType(StrEnum):
    r"""Lattice symmetry types for periodic feature construction.

    Expands the primitive reciprocal basis :math:`\mathbf{b}_\text{base}`
    into an overcomplete set :math:`\mathbf{b} = M \mathbf{b}_\text{base}`
    by applying integer linear combinations. The expanded vectors are used
    by the periodic distance functions (:class:`DistanceType`) to construct
    symmetry-aware features. The primitive basis alone may miss
    symmetry-equivalent directions of the crystal.

    Attributes:
        minimal: Identity matrix (3 vectors). No expansion.
        fcc: Adds :math:`[1,1,1]` combination (4 vectors).
        bcc: Adds :math:`[1,-1,0]`, :math:`[1,0,-1]`,
            :math:`[0,1,-1]` combinations (6 vectors).
        hexagonal: Adds :math:`[-1,-1,0]` combination (4 vectors).
    """

    minimal = "minimal"
    fcc = "fcc"
    bcc = "bcc"
    hexagonal = "hexagonal"


def wrap_positions(positions: jnp.ndarray, lattice: jnp.ndarray) -> jnp.ndarray:
    """Wraps positions into the primary unit cell.

    Args:
        positions: Particle positions with shape (..., ndim).
        lattice: Lattice vectors with shape (ndim, ndim), where each row is a
            lattice vector.

    Returns:
        Wrapped positions with the same shape as ``positions``.
    """
    inv_lattice = jnp.linalg.inv(lattice)
    fractional = positions @ inv_lattice
    wrapped_fractional = fractional % 1.0
    return wrapped_fractional @ lattice


def build_distance_fn(lattice: jnp.ndarray):
    """Computes minimal image distance between particles under PBC.

    Args:
        lattice: Lattice vectors with shape (ndim, ndim).

    Returns:
        A function ``dist_fn(pos_a, pos_b) -> (disp, r)``.
    """
    lattice = jnp.asarray(lattice)
    ndim = lattice.shape[-1]

    # Check diagonal
    ortho_tol = 1e-10
    is_diagonal = jnp.all(
        jnp.abs(lattice - jnp.diag(jnp.diagonal(lattice))) < ortho_tol
    )

    # Check orthogonal
    if is_diagonal:
        is_orthogonal = True
    else:
        is_orthogonal = bool(
            jnp.allclose(
                jnp.triu(lattice @ lattice.T), jnp.zeros((ndim, ndim)), atol=ortho_tol
            )
        )

    # Optimized MIC for diagonal lattices.
    if is_diagonal:
        lat_diag = jnp.diagonal(lattice)

        def distance_fn(ra: jnp.ndarray, rb: jnp.ndarray):
            diff = ra[..., :, None, :] - rb[..., None, :, :]
            disp = (diff + lat_diag / 2) % lat_diag - lat_diag / 2
            dist = jnp.linalg.norm(disp, axis=-1)
            return disp, dist

        return distance_fn

    # Optimized MIC for orthogonal lattices.
    elif is_orthogonal:
        reciprocal_lattice = jnp.linalg.inv(lattice)

        def distance_fn(ra: jnp.ndarray, rb: jnp.ndarray):
            diff = ra[..., :, None, :] - rb[..., None, :, :]
            frac_diff = diff @ reciprocal_lattice
            frac_diff_mic = (frac_diff + 0.5) % 1.0 - 0.5
            disp = frac_diff_mic @ lattice
            dist = jnp.linalg.norm(disp, axis=-1)
            return disp, dist

        return distance_fn

    # General MIC implementation searching 3^ndim neighboring images.
    else:
        mesh_grid = jnp.meshgrid(*[jnp.array([0, 1, 2]) for _ in range(ndim)])
        point_list = jnp.stack([m.ravel() for m in mesh_grid], axis=0).T - 1
        shifts = point_list @ lattice

        def distance_fn(ra: jnp.ndarray, rb: jnp.ndarray):
            diff = ra[..., :, None, :] - rb[..., None, :, :]
            diff_all = diff[..., None, :] + shifts
            dists_all = jnp.linalg.norm(diff_all, axis=-1)
            min_idx = jnp.argmin(dists_all, axis=-1)
            best_disp = jnp.take_along_axis(diff_all, min_idx[..., None, None], axis=-2)
            best_disp = best_disp[..., 0, :]
            best_dist = jnp.linalg.norm(best_disp, axis=-1)
            return best_disp, best_dist

        return distance_fn


def make_pbc_gaussian_proposal(lattice: jnp.ndarray) -> SamplingProposal:
    """Creates a gaussian proposal that wraps positions to the primary cell.

    Args:
        lattice: Lattice vectors with shape (ndim, ndim).

    Returns:
        A sampling proposal function that respects PBC.
    """

    def proposal(rngs: PRNGKey, x, stddev: float | jnp.ndarray):
        x_new = gaussian_proposal(rngs, x, stddev)
        return jax.tree.map(lambda a: wrap_positions(a, lattice), x_new)

    return proposal


def scaled_f(w: jnp.ndarray) -> jnp.ndarray:
    r"""Function f used in polynomial distance (nu_distance).

    .. math::
        f(w) = |w| (1 - |w/\pi|^3 / 4)

    Args:
        w: the fractional coordinates scaled to the range :math:`[-\pi, \pi]`,
            :math:`w = \mathbf{r} \cdot \mathbf{b}`,
            where :math:`\mathbf{b}` is the reciprocal space vectors.

    Returns:
        Scaled value of w.
    """
    return jnp.abs(w) * (1 - jnp.abs(w / jnp.pi) ** 3 / 4.0)


def scaled_g(w: jnp.ndarray) -> jnp.ndarray:
    r"""Function g used in polynomial distance (nu_distance).

    .. math::
        g(w) = w (1 - 1.5 |w/\pi| + 0.5 |w/\pi|^2)

    Args:
        w: the fractional coordinates scaled to the range :math:`[-\pi, \pi]`,
            :math:`w = \mathbf{r} \cdot \mathbf{b}`,
            where :math:`\mathbf{b}` is the reciprocal space vectors.

    Returns:
        Scaled value of w.
    """
    return w * (
        1 - 3.0 / 2.0 * jnp.abs(w / jnp.pi) + 1.0 / 2.0 * jnp.abs(w / jnp.pi) ** 2
    )


def nu_distance(
    xea: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Computes periodic generalized relative and absolute distance using polynomials.

    .. math::
        d(r) = \sqrt{ \sum_i |\mathbf{a}_i|^2 f(\omega_i)^2 + \sum_{i \neq j}
            (\mathbf{a}_i \cdot \mathbf{a}_j) g(\omega_i) g(\omega_j) }

    where :math:`\omega_i` are the fractional coordinates scaled to the range
    :math:`[-\pi, \pi]`.

    Args:
        xea: Relative coordinates between electrons and atoms.
            Shape (..., n_electrons, n_atoms).
        a: (Over complete set of) primitive cell lattice vectors.
        b: (Over complete set of) reciprocal lattice vectors.

    .. math::
        \text{rel} = \sum_i (g(w_i) a_i)

    Returns:
        (:math:`d`, rel) tuple of distances and relative coordinates.

    References:
        - `Phys. Rev. B 94, 035157 (2016) <https://doi.org/10.1103/PhysRevB.94.035157>`_
        - `Nat Commun 13, 1 (2022) <https://doi.org/10.1038/s41467-022-35627-1>`_
    """
    w = jnp.einsum("...ijk,lk->...ijl", xea, b)
    mod = (w + jnp.pi) // (2 * jnp.pi)
    w = w - mod * 2 * jnp.pi
    r1 = (jnp.linalg.norm(a, axis=-1) * scaled_f(w)) ** 2
    sg = scaled_g(w)
    rel = jnp.einsum("...i,ij->...j", sg, a)
    r2 = jnp.einsum("ij,kj->ik", a, a) * (sg[..., :, None] * sg[..., None, :])
    result = jnp.sum(r1, axis=-1) + jnp.sum(
        r2 * (jnp.ones(r2.shape[-2:]) - jnp.eye(r2.shape[-1])), axis=[-1, -2]
    )
    sd = jnp.sqrt(result)
    return sd, rel


def tri_distance(
    xea: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Computes periodic distance using trigonometric functions.

    References:
        - https://doi.org/10.1103/PhysRevLett.130.036401.

    .. math::
        d(r) = \sqrt{ \sum_{i,j} V_{ij} G_{ij} }

    where:

    .. math::
        G_{ij} = \mathbf{a_i} \cdot \mathbf{a_j} \quad (\text{metric tensor})

    .. math::
        V_{ij} = \sin(w_i)\sin(w_j) + (1-\cos(w_i))(1-\cos(w_j))

    .. math::
        w_i = \mathbf{b_i} \cdot \mathbf{r}

    .. math::
        \text{rel} = \sum_i (g(w_i) a_i)

    Returns:
        (sd, rel) tuple of distances and relative coordinates.
    """
    w = jnp.einsum("...ijk,lk->...ijl", xea, b)
    sg = jnp.sin(w)
    cg = jnp.cos(w)
    rel_sin = jnp.einsum("...i,ij->...j", sg, a)
    rel_cos = jnp.einsum("...i,ij->...j", cg, a)
    rel = jnp.concatenate([rel_sin, rel_cos], axis=-1)
    metric = jnp.einsum("ij,kj->ik", a, a)
    vector_sin = sg[..., :, None] * sg[..., None, :]
    vector_cos = (1 - cg[..., :, None]) * (1 - cg[..., None, :])
    vector = vector_cos + vector_sin
    sd = jnp.sqrt(jnp.einsum("...ij,ij->...", vector, metric))
    return sd, rel


def get_distance_function(distance_type: DistanceType):
    """Returns the distance function corresponding to the given DistanceType.

    Args:
        distance_type: Type of periodic distance ('nu' or 'tri').

    Returns:
        Function that computes distance.

    Raises:
        ValueError: If distance_type is unknown.
    """
    if distance_type == DistanceType.nu:
        return nu_distance
    elif distance_type == DistanceType.tri:
        return tri_distance
    else:
        raise ValueError(f"Unrecognized distance function: {distance_type}")


def get_symmetry_lat(
    lattice: jnp.ndarray, sym_type: SymmetryType = SymmetryType.minimal
):
    """Expands reciprocal lattice vectors to include high-symmetry directions.

    This function generates a specific set of reciprocal lattice vectors by applying
    integer linear transformations to the primitive reciprocal basis. This is necessary
    to capture symmetry-equivalent vectors that may not be aligned with the principal
    axes.

    Args:
        lattice: The primitive cell lattice.
        sym_type: The type of symmetry to apply.

    Returns:
        A new Cell object with updated reciprocal_lattice.
    """
    ndim = lattice.shape[-1]
    bv_base = 2 * jnp.pi * jnp.linalg.inv(lattice).T
    if sym_type == SymmetryType.minimal:
        mat = jnp.eye(ndim)
    elif sym_type == SymmetryType.fcc and ndim == 3:
        mat = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    elif sym_type == SymmetryType.bcc and ndim == 3:
        mat = jnp.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1]]
        )
    elif sym_type == SymmetryType.hexagonal and ndim == 3:
        mat = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, 0]])
    else:
        mat = jnp.eye(ndim)

    bv = mat @ bv_base
    av = jnp.linalg.pinv(bv).T
    return av, bv
