# Copyright 2023 Microsoft Corporation
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2026.
#
# Original file was released under MIT, with the full license text
# available at licenses/folx_MIT.txt
#
# This file is distributed under the Apache License 2.0,
# with portions originally licensed under the MIT License.

"""Forward Laplacian rules for reduction and accumulation primitives."""

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from ..guards import is_sparse_laptuple
from ..types import LaplacianHandler, LapTuple
from .core import log_dense_fallback, wrap_general, wrap_linear
from .sparse_ops import sparse_factor_shape, sparse_output_block_axes


def handle_reduce_sum(args, kwargs):
    dense_handler = wrap_linear(jax.lax.reduce_sum_p.bind)
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    axes = kwargs["axes"]
    axes = tuple(range(x.x.ndim)) if axes is None else tuple(axes)
    axes = tuple(axis if axis >= 0 else axis + x.x.ndim for axis in axes)
    y = jax.lax.reduce_sum_p.bind(x.x, **kwargs)
    lapl = jax.lax.reduce_sum_p.bind(x.laplacian, **kwargs)
    owners = x.jacobian.owners.reduce_output_axes(
        axes,
        output_shape=x.jacobian.output_shape,
    )
    if owners is not None:
        blocks = jax.lax.reduce_sum_p.bind(
            x.jacobian.blocks,
            axes=sparse_output_block_axes(axes, output_ndim=x.x.ndim),
        )
        return LapTuple(
            y,
            x.jacobian.with_blocks(
                blocks,
                owners=owners,
            ),
            lapl,
        )
    log_dense_fallback(
        site="reduce_sum",
        kind="unrepresentable",
        reason="reduced owner layout cannot stay sparse",
    )
    jacobian = jax.lax.reduce_sum_p.bind(
        x.jacobian.to_dense(),
        axes=tuple(axis + 1 for axis in axes),
    )
    return LapTuple(y, jacobian, lapl)


def _broadcast_reduced_to_operand(
    reduced: jnp.ndarray,
    operand_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> jnp.ndarray:
    kept_sizes = iter(reduced.shape)
    reshape_shape = tuple(
        1 if axis in axes else next(kept_sizes) for axis in range(len(operand_shape))
    )
    return jnp.broadcast_to(jnp.reshape(reduced, reshape_shape), operand_shape)


def _handle_reduce_select(args, kwargs, *, primitive, op_name: str):
    dense_handler = wrap_linear(primitive.bind)
    if len(args) != 1:
        return dense_handler(args, kwargs)
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)

    axes = kwargs["axes"]
    axes = tuple(range(x.x.ndim)) if axes is None else tuple(axes)
    axes = tuple(axis if axis >= 0 else axis + x.x.ndim for axis in axes)
    y = primitive.bind(x.x, **kwargs)
    y_broadcast = _broadcast_reduced_to_operand(y, x.x.shape, axes)
    mask = x.x == y_broadcast
    tie_count = jax.lax.reduce_sum_p.bind(mask.astype(jnp.int32), axes=axes)
    tie_count = _broadcast_reduced_to_operand(tie_count, x.x.shape, axes)
    weight = mask.astype(x.x.dtype) / tie_count.astype(x.x.dtype)

    blocks = jax.lax.reduce_sum_p.bind(
        x.jacobian.blocks
        * jnp.reshape(weight, sparse_factor_shape(x.jacobian.output_shape)),
        axes=sparse_output_block_axes(axes, output_ndim=x.x.ndim),
    )
    lapl = jax.lax.reduce_sum_p.bind(weight * x.laplacian, axes=axes)
    owners = x.jacobian.owners.reduce_output_axes(
        axes,
        output_shape=x.jacobian.output_shape,
    )
    if owners is not None:
        return LapTuple(
            y,
            x.jacobian.with_blocks(
                blocks,
                owners=owners,
            ),
            lapl,
        )
    log_dense_fallback(
        site=op_name,
        kind="unrepresentable",
        reason="reduced owner layout cannot stay sparse",
    )
    jacobian = jax.lax.reduce_sum_p.bind(
        x.dense_jacobian * weight,
        axes=tuple(axis + 1 for axis in axes),
    )
    return LapTuple(y, jacobian, lapl)


def handle_reduce_max(args, kwargs):
    return _handle_reduce_select(
        args, kwargs, primitive=jax.lax.reduce_max_p, op_name="reduce_max"
    )


def handle_reduce_min(args, kwargs):
    return _handle_reduce_select(
        args, kwargs, primitive=jax.lax.reduce_min_p, op_name="reduce_min"
    )


REDUCTIONS_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.reduce_sum_p: handle_reduce_sum,
    jax.lax.reduce_prod_p: wrap_general(jax.lax.reduce_prod_p.bind),
    jax.lax.reduce_max_p: handle_reduce_max,
    jax.lax.reduce_min_p: handle_reduce_min,
}
if hasattr(jax.lax, "cumsum_p"):
    REDUCTIONS_HANDLERS[jax.lax.cumsum_p] = wrap_linear(jax.lax.cumsum_p.bind)
