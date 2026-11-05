# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""LapNet attention helper and its Forward Laplacian specialization."""

from functools import partial
from typing import assert_never

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.laplacian import (
    ArrayOrLapTuple,
    AutoLaplacianFallback,
    LapTuple,
    Local1Jacobian,
    custom_laplacian,
    is_local1_laptuple,
)
from jaqmc.laplacian.precision import (
    ContractionMode,
    current_contraction_mode,
    scaled_einsum,
)


@custom_laplacian
def lapnet_sparse_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
) -> jnp.ndarray:
    """Compute LapNet attention values before the output projection.

    Returns:
        Attention output with shape `(n_electrons, num_heads, heads_dim)`.
    """
    q, k, v = (x.astype(query.dtype) for x in (query, key, value))
    scale = jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))
    logits = jnp.einsum("ihd,jhd->hij", q, k) / scale
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hij,jhd->ihd", weights, v)


def _as_dense_laptuple(
    arg: ArrayOrLapTuple,
    tracked_size: int,
) -> LapTuple[jnp.ndarray]:
    if isinstance(arg, LapTuple):
        dense = arg.to_dense()
        if dense.dense_jacobian.shape[0] != tracked_size:
            raise ValueError(
                "LapNet attention value has incompatible tracked-input size."
            )
        return dense
    return LapTuple(
        arg,
        jnp.zeros((tracked_size, *arg.shape), dtype=arg.dtype),
        jnp.zeros_like(arg),
    )


def _local1_owner_matches_sparse_axis(
    jacobian: Local1Jacobian,
    *,
    n_electrons: int,
) -> bool:
    owner = jacobian.owners[0]
    return owner.axis == 0 and np.array_equal(
        owner.values, np.arange(n_electrons, dtype=np.int32)
    )


def _supports_local1_attention(
    query: LapTuple[Local1Jacobian],
    key: LapTuple[Local1Jacobian],
) -> bool:
    """Return whether the LapNet Local1 fast path applies."""
    n_electrons = query.x.shape[0]
    return (
        query.shape == key.shape
        and query.jacobian.input_n_particles == n_electrons
        and key.jacobian.input_n_particles == n_electrons
        and query.jacobian.input_coord_dim == key.jacobian.input_coord_dim
        and query.jacobian.owners == key.jacobian.owners
        and _local1_owner_matches_sparse_axis(query.jacobian, n_electrons=n_electrons)
        and _local1_owner_matches_sparse_axis(key.jacobian, n_electrons=n_electrons)
    )


def _attention_quotient(
    numer_x: jnp.ndarray,
    numer_j: jnp.ndarray,
    numer_l: jnp.ndarray,
    denom_x: jnp.ndarray,
    denom_j: jnp.ndarray,
    denom_l: jnp.ndarray,
) -> LapTuple[jnp.ndarray]:
    """Return attention values from dense numerator/denominator derivatives."""
    inv_denom = 1.0 / denom_x
    values_x = numer_x * inv_denom[..., None]
    values_j = numer_j * inv_denom[None, ..., None] - (
        numer_x[None, ...] * denom_j[..., None] * (inv_denom**2)[None, ..., None]
    )
    numer_denom_cross = jnp.sum(numer_j * denom_j[..., None], axis=0)
    denom_grad_sq = jnp.sum(denom_j**2, axis=0)
    values_l = (
        numer_l * inv_denom[..., None]
        - 2 * numer_denom_cross * (inv_denom**2)[..., None]
        - numer_x * denom_l[..., None] * (inv_denom**2)[..., None]
        + 2 * numer_x * denom_grad_sq[..., None] * (inv_denom**3)[..., None]
    )
    values_j = jax.lax.optimization_barrier(values_j)
    values_l = jax.lax.optimization_barrier(values_l)
    values_x = jnp.transpose(values_x, (1, 0, 2))
    values_j = jnp.transpose(values_j, (0, 2, 1, 3))
    values_l = jnp.transpose(values_l, (1, 0, 2))
    return LapTuple(values_x, values_j, values_l)


def _lapnet_sparse_attention_local1_rule(
    query: LapTuple[Local1Jacobian],
    key: LapTuple[Local1Jacobian],
    value: ArrayOrLapTuple,
) -> LapTuple[jnp.ndarray]:
    """Manual LapNet attention rule that keeps ``q`` and ``k`` in Local1 form.

    Returns:
        Dense-output ``LapTuple`` for the attention values.
    """
    n_electrons = query.x.shape[0]
    tracked_size = query.jacobian.input_n_particles * query.jacobian.input_coord_dim
    mode = current_contraction_mode()
    match mode:
        case ContractionMode.STANDARD:
            compute_dtype = query.x.dtype
            primal_prec = None
        case ContractionMode.SCALED_FP16:
            compute_dtype = jnp.float32
            primal_prec = jax.lax.Precision.HIGHEST
        case _:
            assert_never(mode)

    primal_einsum = partial(
        jnp.einsum, precision=primal_prec, preferred_element_type=compute_dtype
    )
    jac_einsum = partial(scaled_einsum, preferred_element_type=compute_dtype)
    primal_matmul = partial(
        jnp.matmul, precision=primal_prec, preferred_element_type=compute_dtype
    )

    v = _as_dense_laptuple(value, tracked_size)
    coord_dim = query.jacobian.input_coord_dim
    eye = jnp.eye(n_electrons, dtype=compute_dtype)

    # Local1 blocks store (support_slot, coord, electron, head, feature).
    q_x = query.x.astype(compute_dtype)
    k_x = key.x.astype(compute_dtype)
    q_l = query.laplacian.astype(compute_dtype)
    k_l = key.laplacian.astype(compute_dtype)
    q_blocks = query.jacobian.blocks[0]
    k_blocks = key.jacobian.blocks[0]

    scale = jnp.sqrt(jnp.asarray(q_x.shape[-1], dtype=compute_dtype))
    logits_x = primal_einsum("ihd,jhd->hij", q_x, k_x) / scale
    logits_grad_i = jac_einsum("cihd,jhd->chij", q_blocks, k_x) / scale
    logits_grad_j = jac_einsum("ihd,cjhd->chij", q_x, k_blocks) / scale
    logits_l = (
        primal_einsum("ihd,jhd->hij", q_l, k_x)
        + primal_einsum("ihd,jhd->hij", q_x, k_l)
    ) / scale
    logits_cross = (2 * jac_einsum("cihd,cihd->chi", q_blocks, k_blocks) / scale).sum(
        axis=0
    )
    logits_l = logits_l + logits_cross[:, :, None] * eye[None, :, :]

    logits_trace = jnp.sum(logits_grad_i**2, axis=0) + jnp.sum(logits_grad_j**2, axis=0)
    logits_trace = logits_trace + (
        2
        * primal_einsum("chij,chij->hij", logits_grad_i, logits_grad_j)
        * eye[None, :, :]
    )

    logits_shift = logits_x - jax.lax.stop_gradient(
        jnp.max(logits_x, axis=-1, keepdims=True)
    )
    weights_x = jnp.exp(logits_shift)
    weights_grad_i = weights_x[None, ...] * logits_grad_i
    weights_grad_j = weights_x[None, ...] * logits_grad_j
    weights_l = weights_x * (logits_l + logits_trace)

    denom_x = jnp.sum(weights_x, axis=-1)
    denom_grad_i = jnp.sum(weights_grad_i, axis=-1)
    denom_grad_j = jnp.transpose(weights_grad_j, (3, 0, 1, 2))
    denom_j = (jnp.einsum("pi,chi->pchi", eye, denom_grad_i) + denom_grad_j).reshape(
        tracked_size, *denom_x.shape
    )
    denom_l = jnp.sum(weights_l, axis=-1)

    v_x = v.x.astype(compute_dtype)
    v_x_h_first = jnp.transpose(v_x, (1, 0, 2))
    v_l_h_first = jnp.transpose(v.laplacian.astype(compute_dtype), (1, 0, 2))
    numer_x = primal_matmul(weights_x, v_x_h_first)

    v_jac_raw = v.jacobian.reshape(n_electrons, coord_dim, *v.x.shape)
    numer_grad_i_local = jac_einsum("chij,hjd->chid", weights_grad_i, v_x_h_first)
    numer_grad_i = jnp.einsum("pi,chid->pchid", eye, numer_grad_i_local)
    numer_grad_j = (
        jnp.transpose(weights_grad_j, (3, 0, 1, 2))[..., None]
        * v_x[:, None, :, None, :]
    )
    numer_grad_v = jac_einsum("hij,pcjhd->pchid", weights_x, v_jac_raw)
    numer_j = (numer_grad_i + numer_grad_j + numer_grad_v).reshape(
        tracked_size, *numer_x.shape
    )

    v_diag_jacobian = v_jac_raw[
        jnp.arange(n_electrons), :, jnp.arange(n_electrons)
    ].astype(compute_dtype)
    v_diag_jacobian_h_first = jnp.transpose(v_diag_jacobian, (1, 2, 0, 3))
    numer_l_cross = (2 * jac_einsum("chij,icjhd->chid", weights_grad_i, v_jac_raw)).sum(
        axis=0
    )
    numer_l = (
        primal_matmul(weights_l, v_x_h_first)
        + primal_matmul(weights_x, v_l_h_first)
        + numer_l_cross
        + 2
        * jnp.sum(
            weights_grad_j[..., None] * v_diag_jacobian_h_first[:, :, None, :, :],
            axis=(0, 3),
        )
    )
    result = _attention_quotient(numer_x, numer_j, numer_l, denom_x, denom_j, denom_l)
    if result.x.dtype != query.x.dtype:
        # SCALED_FP16 accumulates in float32; cast back to the input dtype
        # expected by the surrounding traced computation (handle_custom_laplacian
        # replaces result.x with primal_jaxpr's output).
        return LapTuple(
            result.x.astype(query.x.dtype),
            result.jacobian.astype(query.x.dtype),
            result.laplacian.astype(query.x.dtype),
        )
    return result


@lapnet_sparse_attention.def_laplacian_rule
def _lapnet_sparse_attention_rule(
    query: ArrayOrLapTuple,
    key: ArrayOrLapTuple,
    value: ArrayOrLapTuple,
) -> LapTuple[jnp.ndarray]:
    """Manual Forward Laplacian rule for LapNet attention.

    Returns:
        Dense-output ``LapTuple`` for the attention values.

    Raises:
        AutoLaplacianFallback: query or key is not sparse.
            Keeps as an handler for the dense Forward Laplacian scenario.
    """
    if (
        is_local1_laptuple(query)
        and is_local1_laptuple(key)
        and _supports_local1_attention(query, key)
    ):
        return _lapnet_sparse_attention_local1_rule(query, key, value)
    raise AutoLaplacianFallback("LapNet attention dense boundary")
