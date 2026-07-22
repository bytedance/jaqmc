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

"""Hessian trace (tr(J H J^T)) computation strategies."""

import functools
from collections.abc import Callable

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp

from .ad import hessian as complex_hessian
from .ad import jacrev as complex_jacrev
from .types import (
    ForwardFn,
    LapArgs,
)

# ---------------------------------------------------------------------------
# Core JHJ strategies
# ---------------------------------------------------------------------------


def _flat_wrap(fn: ForwardFn, *x: jnp.ndarray):
    """Return a wrapper that takes a single flattened input vector."""
    _, x_unravel = jfu.ravel_pytree(x)

    def new_fn(flat_x: jnp.ndarray) -> jnp.ndarray:
        x = x_unravel(flat_x)
        return jfu.ravel_pytree(fn(*x))[0]

    return new_fn


def _frobenius_inner(mat1: jnp.ndarray, mat2: jnp.ndarray):
    """Return ``sum_ij mat1_ij * mat2_ij`` over trailing matrix axes."""
    return jnp.einsum("...ij,...ij->...", mat1, mat2)


def _get_reduced_jacobians(*jacobians: jnp.ndarray) -> list[jnp.ndarray]:
    """Reshape each Jacobian to ``(K, D)``.

    Returns:
        Jacobians flattened to a shared leading basis and output columns.
    """
    return [jacobian.reshape(jacobian.shape[0], -1) for jacobian in jacobians]


def JHJ_via_hessian(
    flat_fn: Callable, flat_x: jnp.ndarray, grad_2d: jnp.ndarray
) -> jnp.ndarray:
    """Returns trace(H @ JJ^T) by materializing full Hessian. Used when n > D."""
    flat_hessian = complex_hessian(flat_fn)(flat_x)
    return _frobenius_inner(flat_hessian, grad_2d @ grad_2d.T)


def JHJ_via_trace(
    flat_fn: Callable, flat_x: jnp.ndarray, grad_2d: jnp.ndarray
) -> jnp.ndarray:
    """Returns nested jvp contraction without full Hessian. Used when n <= D."""

    @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
    def vhvp(tangent):
        def vjp(x):
            @functools.partial(jax.vmap, in_axes=(None, -1), out_axes=-1)
            def jvp(x, tangent):
                return jax.jvp(flat_fn, (x,), (tangent,))[1]

            return jvp(x, grad_2d)

        return jax.jvp(vjp, (flat_x,), (tangent,))[1]

    return jnp.trace(vhvp(grad_2d), axis1=-2, axis2=-1)


def JHJ_via_hvp(
    flat_fn: Callable, flat_x: jnp.ndarray, grad_2d: jnp.ndarray
) -> jnp.ndarray:
    """Returns Hessian-vector products without full Hessian.

    For complex-to-real functions.
    """

    @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
    def hvp(tangent):
        return jax.jvp(complex_jacrev(flat_fn), (flat_x,), (tangent,))[1]

    HJ = hvp(grad_2d)
    return _frobenius_inner(HJ, grad_2d)


# ---------------------------------------------------------------------------
# Entry points for specific Hessian patterns
# ---------------------------------------------------------------------------


def general_jac_hessian_jac(fn: ForwardFn, args: LapArgs[jnp.ndarray]):
    """Returns the unraveled Hessian trace.

    Selects via_hessian (n>D) or via_trace (n<=D).
    """
    flat_fn = _flat_wrap(fn, *args.x)
    flat_x = jfu.ravel_pytree(args.x)[0]
    out, unravel = jfu.ravel_pytree(fn(*args.x))

    jacobians = args.jacobian
    grads_2d = _get_reduced_jacobians(*jacobians)
    grad_2d = jnp.concatenate([x.T for x in grads_2d], axis=0)
    jac_dim, inp_dim = grad_2d.shape

    is_complex_to_real = jnp.iscomplexobj(flat_x) and not jnp.iscomplexobj(out)

    if inp_dim > jac_dim:
        if is_complex_to_real:
            # Materializing the Hessian for complex-to-real is not supported.
            # Avoid this by only performing HvJ products.
            flat_out = JHJ_via_hvp(flat_fn, flat_x, grad_2d).real
        else:
            # Works for R→R and C→C thanks to complex-aware jacrev/jacfwd.
            flat_out = JHJ_via_hessian(flat_fn, flat_x, grad_2d)
    else:
        flat_out = JHJ_via_trace(flat_fn, flat_x, grad_2d)
    return unravel(flat_out)


def elementwise_jac_hessian_jac(
    fn: ForwardFn, args: LapArgs[jnp.ndarray]
) -> jnp.ndarray:
    """Return ``tr(J^T H J)`` for unary shape-preserving elementwise functions.

    For ``y = f(x)`` applied independently at each element, the input-space
    Hessian is diagonal, so the Hessian trace reduces pointwise to

    ``f''(x) * sum_k J[k] ** 2``.

    This avoids routing through the generic vmapped Hessian machinery when the
    primitive is already known to be unary and elementwise.
    """
    assert len(args) == 1
    x = args.x[0]
    jacobian = args.jacobian[0]
    ones = jnp.ones_like(x)

    def first_derivative(value):
        return jax.jvp(fn, (value,), (ones,))[1]

    _, second_derivative = jax.jvp(first_derivative, (x,), (ones,))
    return second_derivative * jnp.sum(jacobian * jacobian, axis=0)
