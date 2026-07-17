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

"""Forward Laplacian rules for selection primitives."""

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from ..guards import is_dense_laptuple, is_sparse_laptuple
from ..types import LaplacianHandler, LapTuple
from .core import fallback_dense, wrap_linear
from .sparse_ops import (
    broadcast_sparse_and_plain_to_common_shape,
    broadcast_sparse_laptuple,
    compatible_sparse_metadata,
    scale_sparse_jacobian,
    sparse_factor_shape,
    zero_local_jacobian_like,
)


def _pointwise_select_sparse_and_plain(
    sparse: LapTuple, plain, *, pred, value_fn, sparse_is_lhs: bool
):
    sparse, plain = broadcast_sparse_and_plain_to_common_shape(
        sparse, jnp.asarray(plain)
    )
    mask = pred(sparse.x, plain)
    return LapTuple(
        value_fn(sparse.x, plain) if sparse_is_lhs else value_fn(plain, sparse.x),
        scale_sparse_jacobian(sparse.jacobian, mask.astype(sparse.x.dtype)),
        jnp.where(mask, sparse.laplacian, jnp.zeros_like(sparse.laplacian)),
    )


def _handle_pointwise_select(args, kwargs, *, pred, value_fn, op_name, dense_primitive):
    dense_handler = wrap_linear(dense_primitive.bind)
    lhs, rhs = args
    lhs_sparse = is_sparse_laptuple(lhs)
    rhs_sparse = is_sparse_laptuple(rhs)
    if not lhs_sparse and not rhs_sparse:
        return dense_handler(args, kwargs)
    if any(is_dense_laptuple(arg) for arg in args):
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason=f"{op_name} mixes sparse and dense LapTuple",
        )
    if lhs_sparse ^ rhs_sparse:
        sparse, plain = (lhs, rhs) if lhs_sparse else (rhs, lhs)
        assert isinstance(sparse, LapTuple)
        return _pointwise_select_sparse_and_plain(
            sparse,
            plain,
            pred=pred,
            value_fn=value_fn,
            sparse_is_lhs=lhs_sparse,
        )
    if lhs.jacobian.owners != rhs.jacobian.owners:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="not_implemented",
            reason=f"{op_name} sparse owner layout unsupported",
        )
    out_shape = jnp.broadcast_shapes(
        lhs.jacobian.output_shape,
        rhs.jacobian.output_shape,
    )
    lhs = broadcast_sparse_laptuple(lhs, out_shape)
    rhs = broadcast_sparse_laptuple(rhs, out_shape)
    lhs_wins = pred(lhs.x, rhs.x)
    tied = lhs.x == rhs.x
    dtype = lhs.x.dtype
    weight_lhs = jnp.where(tied, 0.5, lhs_wins.astype(dtype))
    weight_rhs = jnp.where(tied, 0.5, (~lhs_wins & ~tied).astype(dtype))
    mask_lhs = weight_lhs.reshape(sparse_factor_shape(out_shape))
    mask_rhs = weight_rhs.reshape(sparse_factor_shape(out_shape))
    return LapTuple(
        value_fn(lhs.x, rhs.x),
        lhs.jacobian.with_blocks(
            mask_lhs * lhs.jacobian.blocks + mask_rhs * rhs.jacobian.blocks,
            owners=lhs.jacobian.owners,
        ),
        weight_lhs * lhs.laplacian + weight_rhs * rhs.laplacian,
    )


def handle_max(args, kwargs):
    return _handle_pointwise_select(
        args,
        kwargs,
        pred=jnp.greater_equal,
        value_fn=jax.lax.max,
        op_name="max",
        dense_primitive=jax.lax.max_p,
    )


def handle_min(args, kwargs):
    return _handle_pointwise_select(
        args,
        kwargs,
        pred=jnp.less_equal,
        value_fn=jax.lax.min,
        op_name="min",
        dense_primitive=jax.lax.min_p,
    )


def handle_select_n(args, kwargs):
    """Strip derivatives from the selector and propagate the chosen case.

    Returns:
        The selected value with derivatives propagated from the chosen case, or
        a dense fallback result when sparse propagation is not representable.
    """
    dense_handler = wrap_linear(jax.lax.select_n_p.bind)
    which = args[0]
    if isinstance(which, LapTuple):
        which = which.x
    cases = args[1:]
    if all(not isinstance(case, LapTuple) for case in cases):
        return dense_handler((which, *cases), kwargs)
    if any(is_dense_laptuple(case) for case in cases):
        return fallback_dense(
            dense_handler,
            (which, *cases),
            kwargs,
            kind="unrepresentable",
            reason="select_n mixes sparse and dense LapTuple",
        )

    out_shape = jnp.broadcast_shapes(
        *(jnp.shape(case.x if isinstance(case, LapTuple) else case) for case in cases)
    )
    broadcasted_cases = []
    sparse_template = None
    for case in cases:
        if isinstance(case, LapTuple):
            if sparse_template is None:
                sparse_template = case
            elif (
                type(case.jacobian) is not type(sparse_template.jacobian)
                or case.jacobian.owners != sparse_template.jacobian.owners
                or not compatible_sparse_metadata(
                    case.jacobian,
                    sparse_template.jacobian,
                )
            ):
                return fallback_dense(
                    dense_handler,
                    (which, *cases),
                    kwargs,
                    kind="not_implemented",
                    reason="select_n mixed sparse operand families",
                )
            broadcasted_cases.append(broadcast_sparse_laptuple(case, out_shape))
            continue
        broadcasted_cases.append(jnp.broadcast_to(case, out_shape))

    assert sparse_template is not None
    zeros = zero_local_jacobian_like(sparse_template.jacobian, out_shape)
    sparse_cases = [
        case
        if isinstance(case, LapTuple)
        else LapTuple(case, zeros, jnp.zeros_like(case))
        for case in broadcasted_cases
    ]
    clipped_which = jnp.clip(which, 0, len(sparse_cases) - 1)

    def select_case(selected: LapTuple, case: LapTuple, picks_case) -> LapTuple:
        mask = jnp.broadcast_to(picks_case, selected.x.shape).reshape(
            sparse_factor_shape(selected.x.shape)
        )
        return LapTuple(
            jnp.where(picks_case, case.x, selected.x),
            selected.jacobian.with_blocks(
                jnp.where(
                    mask,
                    case.jacobian.blocks,
                    selected.jacobian.blocks,
                ),
                owners=selected.jacobian.owners,
            ),
            jnp.where(picks_case, case.laplacian, selected.laplacian),
        )

    selected = sparse_cases[0]
    for index, case in enumerate(sparse_cases[1:], start=1):
        selected = select_case(selected, case, clipped_which == index)
    return selected


SELECTION_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.max_p: handle_max,
    jax.lax.min_p: handle_min,
    jax.lax.select_n_p: handle_select_n,
}
