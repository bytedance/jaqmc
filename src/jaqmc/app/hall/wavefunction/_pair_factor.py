# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Reduced log pair factors and a memory-efficient Forward Laplacian rule."""

import math

import numpy as np
from jax import numpy as jnp

from jaqmc.laplacian import (
    ArrayOrLapTuple,
    AutoLaplacianFallback,
    LapTuple,
    custom_laplacian,
)
from jaqmc.laplacian.sparse import Local1Jacobian

# Haar RMS of the pair spinor a_ij = u_i v_j - v_i u_j: |a_ij|^2 =
# (1 - cos theta_ij) / 2 and cos theta_ij is uniform on [-1, 1], so
# E|a_ij|^2 = 1/2. Unit-RMS factors keep high-flux sectors from being
# exponentially suppressed in coherent multi-sector sums.
_HAAR_RMS = 1 / math.sqrt(2)


@custom_laplacian
def log_pair_factor(w: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    r"""Reduce pair factors :math:`G_{ij} = w_i \cdot z_j`.

    ``w`` has shape ``(nelec, ndets, channels, norb, 2)`` and ``z`` has shape
    ``(nelec, 2)``. Inside Forward Laplacian, the custom rule applies when each
    ``z_j`` depends only on electron ``j``'s own coordinates; ``w`` may depend
    on every electron.

    Returns:
        Log pair factors with shape ``(ndets, norb, nelec)``.
    """
    nelec = z.shape[0]
    pair_bilinear = jnp.einsum("idcka,ja->dckij", w, z) / _HAAR_RMS
    off_diagonal = 1.0 - jnp.eye(nelec)
    pair_bilinear = pair_bilinear * off_diagonal + (1.0 - off_diagonal)
    return jnp.sum(jnp.log(pair_bilinear), axis=(1, 4))


@log_pair_factor.def_laplacian_rule
def _log_pair_factor_rule(
    w: ArrayOrLapTuple,
    z: ArrayOrLapTuple,
) -> LapTuple[jnp.ndarray]:
    r"""Reduce pair derivatives using the per-electron sparsity of ``z``.

    For :math:`G_{ij} = w_i \cdot z_j`, the derivative along coordinate
    :math:`s` of electron :math:`m` is

    .. math::

        \partial_{ms} G_{ij} = \partial_{ms} w_i \cdot z_j
        + \delta_{mj}\, w_i \cdot \partial_{js} z_j,

    because ``z_j`` depends only on electron ``j``. With
    :math:`\Delta\log G = \Delta G / G - \sum_p (\partial_p G)^2 / G^2`, every
    term containing a derivative of ``z`` only needs electron ``j``'s own
    coordinates, and the all-coordinate square of the ``w`` term collapses to
    :math:`z_j^\mathsf{T} W_i z_j` with the per-electron Gram matrix
    :math:`W_i = \sum_p \partial_p w_i\, \partial_p w_i^\mathsf{T}`. Every
    intermediate stays ``O(N^3)``.

    Returns:
        Reduced pair-factor values, Jacobians, and Laplacians.

    Raises:
        AutoLaplacianFallback: An input is untracked, ``z`` lacks per-electron
            sparsity, or other inputs are tracked alongside ``z``.
    """
    if not (isinstance(w, LapTuple) and isinstance(z, LapTuple)):
        raise AutoLaplacianFallback("pair factor with untracked input")
    z_jac = z.jacobian
    nelec = z.x.shape[0]
    if not (
        isinstance(z_jac, Local1Jacobian)
        and z_jac.input_owner_axis == 0
        and z_jac.input_n_particles == nelec
        and z_jac.output_shape == z.x.shape
        and z_jac.owners.roles[0].axis == 0
        and np.array_equal(z_jac.owners.roles[0].values, np.arange(nelec))
    ):
        raise AutoLaplacianFallback("pair factor spinor is not per-electron sparse")
    # dz[s, j]: derivative of z_j along coordinate s of electron j.
    dz = z_jac.blocks[0]
    ncoord = dz.shape[0]
    w = w.to_dense()
    if w.jacobian.shape[0] != nelec * ncoord:
        raise AutoLaplacianFallback("pair factor with extra tracked inputs")
    # dw[m, s]: derivative along coordinate s of electron m.
    dw = w.jacobian.reshape(nelec, ncoord, *w.x.shape)

    pair = jnp.einsum("idcka,ja->dckij", w.x, z.x)
    active = (1.0 - jnp.eye(nelec))[None, None, None]
    inverse_pair = jnp.where(active, 1.0 / pair, 0.0)
    inverse_pair_squared = inverse_pair**2

    # ∂G along electron j's own coordinates: from z_j, and from w_i.
    own_from_z = jnp.einsum("idcka,sja->sdckij", w.x, dz)
    own_from_w = jnp.einsum("jsidcka,ja->sdckij", dw, z.x)

    weighted_z = jnp.einsum("dckij,ja->idcka", inverse_pair, z.x)
    log_jac = jnp.einsum("msidcka,idcka->msdki", dw, weighted_z) + jnp.einsum(
        "sdckij,dckij->jsdki", own_from_z, inverse_pair
    )
    log_jac = log_jac.reshape(nelec * ncoord, *log_jac.shape[2:])

    pair_laplacian = (
        jnp.einsum("idcka,ja->dckij", w.laplacian, z.x)
        + jnp.einsum("idcka,ja->dckij", w.x, z.laplacian)
        + 2 * jnp.einsum("jsidcka,sja->dckij", dw, dz)
    )
    w_gram = jnp.einsum("msidcka,msidckb->idckab", dw, dw)
    squared_gradient = jnp.einsum("idckab,ja,jb->dckij", w_gram, z.x, z.x) + jnp.sum(
        2 * own_from_w * own_from_z + own_from_z**2, axis=0
    )
    log_lapl = jnp.einsum(
        "dckij,dckij->dki", pair_laplacian, inverse_pair
    ) - jnp.einsum("dckij,dckij->dki", squared_gradient, inverse_pair_squared)
    return LapTuple(log_pair_factor(w.x, z.x), log_jac, log_lapl)
