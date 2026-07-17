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

"""Forward Laplacian rules for arithmetic primitives."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from optax.tree_utils import tree_add

from ..guards import (
    is_dense_laptuple,
    is_laptuple,
    is_local1_laptuple,
    is_local2_laptuple,
    is_sparse_laptuple,
    lap_args_are_dense,
)
from ..jvp import dense_jvp
from ..sparse import (
    Local1Jacobian,
    Local2Jacobian,
    SparseJacobian,
)
from ..types import LaplacianHandler, LapTuple
from .core import (
    broadcast_dense_jacobian,
    fallback_dense,
    pack_laptuple,
    setup_handler,
    wrap_general,
    wrap_linear,
    wrap_multiplication,
)
from .sparse_ops import (
    broadcast_sparse_and_plain_to_common_shape,
    broadcast_sparse_jacobian,
    broadcast_sparse_laptuple,
    lift_local1_binary_to_local2,
    owner_ids_equal_mask,
    require_compatible_sparse_metadata,
    scale_sparse_jacobian,
    sparse_trace_jac_jacT,
)


def _add_sub_local1(
    lhs,
    rhs,
    *,
    subtract_rhs: bool,
):
    lhs_jac = lhs.jacobian
    rhs_jac = rhs.jacobian
    assert isinstance(lhs_jac, Local1Jacobian)
    assert isinstance(rhs_jac, Local1Jacobian)
    require_compatible_sparse_metadata(lhs_jac, rhs_jac, operation="add/sub")
    out_shape = jnp.broadcast_shapes(
        lhs_jac.output_shape,
        rhs_jac.output_shape,
    )
    lhs_jac = broadcast_sparse_jacobian(lhs_jac, out_shape)
    rhs_jac = broadcast_sparse_jacobian(rhs_jac, out_shape)
    rhs_lapl = -rhs.laplacian if subtract_rhs else rhs.laplacian
    if lhs_jac.owners != rhs_jac.owners:
        # A sum of two Local1 states with different owners depends on one owner
        # from each addend, so the exact sparse closure is Local2 rather than Local1.
        return LapTuple(
            jax.lax.sub(lhs.x, rhs.x) if subtract_rhs else jax.lax.add(lhs.x, rhs.x),
            lift_local1_binary_to_local2(
                lhs_jac,
                rhs_jac,
                rhs_sign=-1 if subtract_rhs else 1,
            ),
            lhs.laplacian + rhs_lapl,
        )
    rhs_blocks = -rhs_jac.blocks if subtract_rhs else rhs_jac.blocks
    return LapTuple(
        jax.lax.sub(lhs.x, rhs.x) if subtract_rhs else jax.lax.add(lhs.x, rhs.x),
        lhs_jac.with_blocks(lhs_jac.blocks + rhs_blocks, owners=lhs_jac.owners),
        lhs.laplacian + rhs_lapl,
    )


def _add_sub_local2(
    lhs,
    rhs,
    *,
    subtract_rhs: bool,
):
    lhs_jac = lhs.jacobian
    rhs_jac = rhs.jacobian
    assert isinstance(lhs_jac, Local2Jacobian)
    assert isinstance(rhs_jac, Local2Jacobian)
    require_compatible_sparse_metadata(lhs_jac, rhs_jac, operation="add/sub")
    out_shape = jnp.broadcast_shapes(
        lhs_jac.output_shape,
        rhs_jac.output_shape,
    )
    lhs_jac = broadcast_sparse_jacobian(lhs_jac, out_shape)
    rhs_jac = broadcast_sparse_jacobian(rhs_jac, out_shape)
    if lhs_jac.owners != rhs_jac.owners:
        # Local2 + Local2 stays sparse only when both support slots line up
        # role-by-role; otherwise the result would need a richer owner model.
        return None
    rhs_blocks = -rhs_jac.blocks if subtract_rhs else rhs_jac.blocks
    rhs_lapl = -rhs.laplacian if subtract_rhs else rhs.laplacian
    return LapTuple(
        jax.lax.sub(lhs.x, rhs.x) if subtract_rhs else jax.lax.add(lhs.x, rhs.x),
        lhs_jac.with_blocks(lhs_jac.blocks + rhs_blocks, owners=lhs_jac.owners),
        lhs.laplacian + rhs_lapl,
    )


def _add_sub_handler(args, kwargs, *, subtract_rhs: bool):
    primitive = jax.lax.sub_p.bind if subtract_rhs else jax.lax.add_p.bind
    dense_handler = wrap_linear(primitive)
    lhs, rhs = args
    lhs_sparse = is_sparse_laptuple(lhs)
    rhs_sparse = is_sparse_laptuple(rhs)
    if not lhs_sparse and not rhs_sparse:
        return dense_handler(args, kwargs)
    if lhs_sparse and rhs_sparse:
        if is_local1_laptuple(lhs) and is_local1_laptuple(rhs):
            return _add_sub_local1(lhs, rhs, subtract_rhs=subtract_rhs)
        if is_local2_laptuple(lhs) and is_local2_laptuple(rhs):
            result = _add_sub_local2(lhs, rhs, subtract_rhs=subtract_rhs)
            if result is not None:
                return result
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="aggregate add/sub unsupported",
        )

    sparse = lhs if lhs_sparse else rhs
    plain = rhs if lhs_sparse else lhs
    if isinstance(plain, LapTuple):
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="add/sub sparse with dense LapTuple",
        )
    if not is_sparse_laptuple(sparse):
        msg = "internal error: expected LapTuple with sparse Jacobian payload"
        raise RuntimeError(msg)
    sparse, plain = broadcast_sparse_and_plain_to_common_shape(sparse, plain)
    sparse_jac: SparseJacobian = sparse.jacobian
    sparse_lapl = sparse.laplacian
    if lhs_sparse:
        value = (
            jax.lax.sub(sparse.x, plain)
            if subtract_rhs
            else jax.lax.add(sparse.x, plain)
        )
        lapl = sparse_lapl
    elif subtract_rhs:
        value = jax.lax.sub(plain, sparse.x)
        sparse_jac = scale_sparse_jacobian(
            sparse_jac, jnp.array(-1, dtype=sparse.x.dtype)
        )
        lapl = -sparse_lapl
    else:
        value = jax.lax.add(plain, sparse.x)
        lapl = sparse_lapl
    return LapTuple(value, sparse_jac, lapl)


def handle_add(args, kwargs):
    return _add_sub_handler(args, kwargs, subtract_rhs=False)


def handle_sub(args, kwargs):
    return _add_sub_handler(args, kwargs, subtract_rhs=True)


def handle_neg(args, kwargs):
    del kwargs
    x = args[0]
    if not is_sparse_laptuple(x):
        return wrap_linear(jax.lax.neg_p.bind)(args, {})
    return LapTuple(
        -x.x,
        scale_sparse_jacobian(x.jacobian, jnp.array(-1, dtype=x.x.dtype)),
        -x.laplacian,
    )


def handle_rem(args, kwargs):
    dense_handler = wrap_linear(jax.lax.rem_p.bind)
    lhs, rhs = args
    lhs_sparse = is_sparse_laptuple(lhs)
    rhs_sparse = is_sparse_laptuple(rhs)
    if not lhs_sparse and not rhs_sparse:
        return dense_handler(args, kwargs)
    if lhs_sparse and not isinstance(rhs, LapTuple):
        if not is_sparse_laptuple(lhs):
            msg = "internal error: expected sparse LapTuple remainder dividend"
            raise RuntimeError(msg)
        sparse, plain = broadcast_sparse_and_plain_to_common_shape(
            lhs, jnp.asarray(rhs)
        )
        return LapTuple(jax.lax.rem(sparse.x, plain), sparse.jacobian, sparse.laplacian)
    return fallback_dense(
        dense_handler,
        args,
        kwargs,
        kind="unrepresentable",
        reason="rem tracked divisor unsupported",
    )


def _mul_local1_local1(
    lhs: LapTuple[Local1Jacobian], rhs: LapTuple[Local1Jacobian]
) -> LapTuple[Local1Jacobian | Local2Jacobian]:
    require_compatible_sparse_metadata(lhs.jacobian, rhs.jacobian, operation="mul")
    out_shape = jnp.broadcast_shapes(
        lhs.jacobian.output_shape,
        rhs.jacobian.output_shape,
    )
    lhs = broadcast_sparse_laptuple(lhs, out_shape)
    rhs = broadcast_sparse_laptuple(rhs, out_shape)
    y = lhs.x * rhs.x
    lapl = rhs.x * lhs.laplacian + lhs.x * rhs.laplacian
    # Product rule cross term 2 tr(J_lhs J_rhs^T). With matching owners it stays
    # in the same Local1 slot; otherwise the exact derivative depends on two
    # owner slots and must be lifted to Local2.
    cross_trace = jnp.sum(lhs.jacobian.blocks * rhs.jacobian.blocks, axis=(0, 1))
    lhs_scaled = scale_sparse_jacobian(lhs.jacobian, rhs.x)
    rhs_scaled = scale_sparse_jacobian(rhs.jacobian, lhs.x)
    if lhs.jacobian.owners == rhs.jacobian.owners:
        return LapTuple(
            y,
            lhs_scaled.with_blocks(
                lhs_scaled.blocks + rhs_scaled.blocks,
                owners=lhs_scaled.owners,
            ),
            lapl + 2 * cross_trace,
        )
    # Mixed second-derivative contributions only survive where the two Local1
    # supports refer to the same original owner.
    same_owner = owner_ids_equal_mask(
        lhs.jacobian.owners[0],
        rhs.jacobian.owners[0],
        y.shape,
    )
    return LapTuple(
        y,
        lift_local1_binary_to_local2(lhs_scaled, rhs_scaled),
        lapl + 2 * jnp.where(same_owner, cross_trace, 0),
    )


def _mul_local2_local2(
    lhs: LapTuple[Local2Jacobian], rhs: LapTuple[Local2Jacobian]
) -> LapTuple[Local2Jacobian] | None:
    require_compatible_sparse_metadata(lhs.jacobian, rhs.jacobian, operation="mul")
    out_shape = jnp.broadcast_shapes(
        lhs.jacobian.output_shape,
        rhs.jacobian.output_shape,
    )
    lhs = broadcast_sparse_laptuple(lhs, out_shape)
    rhs = broadcast_sparse_laptuple(rhs, out_shape)
    if lhs.jacobian.owners != rhs.jacobian.owners:
        return None
    y = lhs.x * rhs.x
    lhs_scaled = scale_sparse_jacobian(lhs.jacobian, rhs.x)
    rhs_scaled = scale_sparse_jacobian(rhs.jacobian, lhs.x)
    same_owner = owner_ids_equal_mask(
        lhs.jacobian.owners[0],
        lhs.jacobian.owners[1],
        y.shape,
    )
    # tr(J_lhs J_rhs^T) splits into same-slot terms plus cross-slot terms.
    # The cross-slot terms are valid only where both support slots refer to the
    # same owner pair, so they are masked by the owner-equality test.
    diag = jnp.sum(lhs.jacobian.blocks[0] * rhs.jacobian.blocks[0], axis=0) + jnp.sum(
        lhs.jacobian.blocks[1] * rhs.jacobian.blocks[1], axis=0
    )
    offdiag = jnp.sum(
        lhs.jacobian.blocks[0] * rhs.jacobian.blocks[1], axis=0
    ) + jnp.sum(lhs.jacobian.blocks[1] * rhs.jacobian.blocks[0], axis=0)
    cross_trace = diag + jnp.where(same_owner, offdiag, 0)
    return LapTuple(
        y,
        lhs_scaled.with_blocks(
            lhs_scaled.blocks + rhs_scaled.blocks,
            owners=lhs_scaled.owners,
        ),
        rhs.x * lhs.laplacian + lhs.x * rhs.laplacian + 2 * cross_trace,
    )


def handle_mul(args, kwargs):
    dense_handler = wrap_multiplication(jax.lax.mul)
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
            reason="mul sparse with dense LapTuple",
        )

    if lhs_sparse ^ rhs_sparse:
        sparse, plain = (lhs, rhs) if lhs_sparse else (rhs, lhs)
        assert isinstance(sparse, LapTuple)
        assert not isinstance(plain, LapTuple)
        sparse, plain = broadcast_sparse_and_plain_to_common_shape(
            sparse, jnp.asarray(plain)
        )
        return LapTuple(
            sparse.x * plain,
            scale_sparse_jacobian(sparse.jacobian, plain),
            sparse.laplacian * plain,
        )

    if is_local1_laptuple(lhs) and is_local1_laptuple(rhs):
        return _mul_local1_local1(lhs, rhs)
    if is_local2_laptuple(lhs) and is_local2_laptuple(rhs):
        result = _mul_local2_local2(lhs, rhs)
        if result is not None:
            return result
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="mul sparse owner layout unsupported",
        )
    return fallback_dense(
        dense_handler,
        args,
        kwargs,
        kind="not_implemented",
        reason="mul mixed sparse operand families",
    )


def handle_pow(args, kwargs):
    dense_handler = wrap_general(jax.lax.pow_p.bind)
    lhs, rhs = args
    lhs_sparse = is_sparse_laptuple(lhs)
    rhs_sparse = is_sparse_laptuple(rhs)
    if not lhs_sparse and not rhs_sparse:
        return dense_handler(args, kwargs)
    if lhs_sparse and not is_laptuple(rhs):
        base, exponent = broadcast_sparse_and_plain_to_common_shape(lhs, rhs)
        flat_base = base.x.reshape(-1)
        flat_exponent = exponent.reshape(-1)

        def derivative_coefficients(base, exponent):
            tangent = jnp.ones_like(base)
            scalar_fn = partial(jax.lax.pow, y=exponent)
            return jax.jvp(
                lambda v: jax.jvp(scalar_fn, (v,), (tangent,))[1], (base,), (tangent,)
            )

        first, second = jax.vmap(derivative_coefficients)(flat_base, flat_exponent)
        first = first.reshape(base.x.shape)
        second = second.reshape(base.x.shape)
        return LapTuple(
            jax.lax.pow(base.x, exponent),
            scale_sparse_jacobian(base.jacobian, first),
            first * base.laplacian + second * sparse_trace_jac_jacT(base.jacobian),
        )
    return fallback_dense(
        dense_handler,
        args,
        kwargs,
        kind="unrepresentable",
        reason="pow tracked exponent unsupported",
    )


def _sparse_reciprocal(x: LapTuple) -> LapTuple:
    inv_x = jnp.reciprocal(x.x)
    inv_x_sq = inv_x * inv_x
    inv_x_cu = inv_x_sq * inv_x
    # Reciprocal identities:
    #   grad(1 / x) = -(1 / x^2) grad(x)
    #   Delta(1 / x) = -(Delta x) / x^2 + 2 tr(J J^T) / x^3.
    return LapTuple(
        inv_x,
        scale_sparse_jacobian(x.jacobian, -inv_x_sq),
        -x.laplacian * inv_x_sq + 2 * sparse_trace_jac_jacT(x.jacobian) * inv_x_cu,
    )


def handle_div(args, kwargs):
    # Keep the common sparse cases analytical: sparse/plain division stays sparse,
    # and a sparse denominator can be rewritten as reciprocal times multiply.
    if any(is_sparse_laptuple(arg) for arg in args) and any(
        is_dense_laptuple(arg) for arg in args
    ):
        return fallback_dense(
            handle_div,
            args,
            kwargs,
            kind="unrepresentable",
            reason="div mixes sparse and dense LapTuple",
        )
    lhs, rhs = args
    if is_sparse_laptuple(lhs) and not isinstance(rhs, LapTuple):
        sparse, plain = broadcast_sparse_and_plain_to_common_shape(lhs, rhs)
        inv_plain = jnp.reciprocal(plain)
        return LapTuple(
            sparse.x * inv_plain,
            scale_sparse_jacobian(sparse.jacobian, inv_plain),
            sparse.laplacian * inv_plain,
        )
    if is_sparse_laptuple(rhs):
        return handle_mul((lhs, _sparse_reciprocal(rhs)), {})
    if any(is_sparse_laptuple(arg) for arg in args):
        return fallback_dense(
            handle_div,
            args,
            kwargs,
            kind="not_implemented",
            reason="div sparse transform unsupported",
        )
    merged_fwd, lapl_args = setup_handler(jax.lax.div, args, kwargs)
    if lapl_args is None:
        return merged_fwd()

    if not lap_args_are_dense(lapl_args):
        msg = "internal error: expected LapArgs with dense Jacobian arrays"
        raise RuntimeError(msg)
    dense_args = lapl_args

    x_val, y_val = tuple(a.x if isinstance(a, LapTuple) else a for a in args)
    out_shape = jnp.broadcast_shapes(x_val.shape, y_val.shape)
    strategy = "elementwise"
    if len(dense_args) == 1 and dense_args.x[0].shape != out_shape:
        strategy = "split"
    y, grad_y, lapl_y = dense_jvp(merged_fwd, dense_args, in_axes=(), strategy=strategy)

    if len(dense_args) == 1 and dense_args.x[0] is x_val:
        return pack_laptuple(y, grad_y, lapl_y)

    numerator = jnp.broadcast_to(x_val, out_shape)
    denominator = jnp.broadcast_to(y_val, out_shape)
    denominator_jac = broadcast_dense_jacobian(dense_args.jacobian[-1], out_shape)
    hessian = (
        2
        * numerator
        / denominator**3
        * jnp.sum(denominator_jac * denominator_jac, axis=0)
    )
    if len(dense_args) == 2:
        lhs_jac = broadcast_dense_jacobian(dense_args.jacobian[0], out_shape)
        hessian -= 2 / denominator**2 * jnp.sum(lhs_jac * denominator_jac, axis=0)

    return pack_laptuple(y, grad_y, tree_add(lapl_y, hessian))


ARITHMETIC_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.add_p: handle_add,
    jax.lax.sub_p: handle_sub,
    jax.lax.neg_p: handle_neg,
    jax.lax.mul_p: handle_mul,
    jax.lax.pow_p: handle_pow,
    jax.lax.div_p: handle_div,
    jax.lax.rem_p: handle_rem,
}
