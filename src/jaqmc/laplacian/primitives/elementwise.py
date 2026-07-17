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

"""Forward Laplacian rules for elementwise primitives."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from ..guards import is_sparse_laptuple
from ..types import (
    ArrayOrLapTuple,
    LaplacianHandler,
    LapTuple,
)
from .core import (
    wrap_elementwise,
    wrap_general,
    wrap_linear,
    wrap_without_fwd_laplacian,
)
from .sparse_ops import (
    scale_sparse_jacobian,
    sparse_factor_shape,
    sparse_trace_jac_jacH,
    sparse_trace_jac_jacT,
)


def make_unary_elementwise_sparse_handler(fn) -> Any:
    dense_handler = wrap_elementwise(fn)

    def first_and_second_derivatives(
        value: jnp.ndarray,
        kwargs: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        tangent = jnp.ones_like(value)
        scalar_fn = partial(fn, **kwargs)
        return jax.jvp(
            lambda v: jax.jvp(scalar_fn, (v,), (tangent,))[1], (value,), (tangent,)
        )

    def handler(
        args: tuple[ArrayOrLapTuple, ...],
        kwargs: dict[str, Any],
    ) -> ArrayOrLapTuple:
        x = args[0]
        if not is_sparse_laptuple(x):
            return dense_handler(args, kwargs)
        flat_x = x.x.reshape(-1)
        first, second = jax.vmap(
            lambda value: first_and_second_derivatives(value, kwargs)
        )(flat_x)
        first = first.reshape(x.x.shape)
        second = second.reshape(x.x.shape)
        jacobian = scale_sparse_jacobian(x.jacobian, first)
        lapl = first * x.laplacian + second * sparse_trace_jac_jacT(x.jacobian)
        return LapTuple(fn(x.x, **kwargs), jacobian, lapl)

    return handler


def handle_abs(
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
) -> ArrayOrLapTuple:
    x = args[0]
    val = x.x if isinstance(x, LapTuple) else x
    if jnp.iscomplexobj(val):
        if not isinstance(x, LapTuple):
            return jnp.abs(val)

        y = jnp.abs(x.x)
        # For z = a + ib, d|z| projects dz onto the radial direction z / |z|.
        # The Laplacian also receives the transverse curvature term
        # (tr(J J^H) - tr(J_radial J_radial^T)) / |z|.
        lapl = (x.x.real * x.laplacian.real + x.x.imag * x.laplacian.imag) / y
        if is_sparse_laptuple(x):
            sparse_jacobian = x.jacobian.with_blocks(
                (
                    x.x.real.reshape(sparse_factor_shape(y.shape))
                    * x.jacobian.blocks.real
                    + x.x.imag.reshape(sparse_factor_shape(y.shape))
                    * x.jacobian.blocks.imag
                )
                / y.reshape(sparse_factor_shape(y.shape))
            )
            lapl += (
                sparse_trace_jac_jacH(x.jacobian)
                - sparse_trace_jac_jacT(sparse_jacobian)
            ) / y
            return LapTuple(
                y,
                sparse_jacobian.astype(y.dtype),
                lapl.astype(y.dtype),
            )
        dense_jacobian = x.dense_jacobian
        radial_jacobian = (
            x.x.real * dense_jacobian.real + x.x.imag * dense_jacobian.imag
        ) / y
        lapl += (
            jnp.sum(jnp.abs(dense_jacobian) ** 2, axis=0)
            - jnp.sum(radial_jacobian**2, axis=0)
        ) / y
        return LapTuple(y, radial_jacobian.astype(y.dtype), lapl.astype(y.dtype))
    if is_sparse_laptuple(x):
        # Away from zero, abs is multiplication by the sign mask, so the sparse
        # rule just scales Jacobian and Laplacian pointwise.
        factor = jnp.where(x.x >= 0, jnp.ones_like(x.x), -jnp.ones_like(x.x))
        return LapTuple(
            jax.lax.abs(x.x),
            scale_sparse_jacobian(x.jacobian, factor),
            factor * x.laplacian,
        )
    return wrap_linear(jax.lax.abs)(args, kwargs)


def handle_round(
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
) -> ArrayOrLapTuple:
    dense_handler = wrap_linear(jax.lax.round_p.bind)
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    zeros = x.jacobian.with_blocks(jnp.zeros_like(x.jacobian.blocks))
    return LapTuple(
        jax.lax.round_p.bind(x.x, **kwargs),
        zeros,
        jnp.zeros_like(x.x),
    )


ELEMENTWISE_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    p: make_unary_elementwise_sparse_handler(p.bind)
    for p in [
        jax.lax.exp_p,
        jax.lax.log_p,
        jax.lax.sin_p,
        jax.lax.cos_p,
        jax.lax.sqrt_p,
        jax.lax.integer_pow_p,
        jax.lax.logistic_p,
        jax.lax.rsqrt_p,
        jax.lax.log1p_p,
        jax.lax.expm1_p,
        jax.lax.tan_p,
        jax.lax.asin_p,
        jax.lax.acos_p,
        jax.lax.atan_p,
        jax.lax.tanh_p,
    ]
}
ELEMENTWISE_HANDLERS.update(
    {
        jax.lax.abs_p: handle_abs,
        jax.lax.atan2_p: wrap_general(jax.lax.atan2_p.bind),
        jax.lax.round_p: handle_round,
        jax.lax.sign_p: wrap_without_fwd_laplacian(jax.lax.sign),
        "logaddexp": wrap_general(jnp.logaddexp),
        # The following functions are JIT-ted
        "softplus": make_unary_elementwise_sparse_handler(jax.nn.softplus),
        "silu": make_unary_elementwise_sparse_handler(jax.nn.silu),
    }
)
if hasattr(jax.lax, "square_p"):
    ELEMENTWISE_HANDLERS[jax.lax.square_p] = make_unary_elementwise_sparse_handler(
        jax.lax.square_p.bind
    )
