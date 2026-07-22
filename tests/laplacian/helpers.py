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

"""Shared test utilities for Forward Laplacian tests."""

import operator
from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaqmc.laplacian import LapTuple, forward_laplacian, make_laplacian_input


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-10):
    """Assert arrays share shape and are numerically close."""
    assert jnp.shape(actual) == jnp.shape(expected), (
        f"Shape mismatch:\n  actual:   {jnp.shape(actual)}\n"
        f"  expected: {jnp.shape(expected)}"
    )
    assert jnp.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Not close:\n  actual:   {actual}\n  expected: {expected}\n"
        f"  max diff: {jnp.max(jnp.abs(actual - expected))}"
    )


def assert_laptuple_allclose(actual, expected, rtol=1e-5, atol=1e-10):
    """Assert two LapTuples agree on value, dense Jacobian, and Laplacian."""
    assert_allclose(actual.x, expected.x, rtol=rtol, atol=atol)
    assert_allclose(
        actual.dense_jacobian, expected.dense_jacobian, rtol=rtol, atol=atol
    )
    assert_allclose(actual.laplacian, expected.laplacian, rtol=rtol, atol=atol)


def brute_force_oracle(
    fn: Callable, *args, expected_fn: Callable | None = None
) -> LapTuple:
    """Return the dense chain-rule oracle for ``fn(*args)``.

    Plain-array calls cover simple top-level auto-seeding. Dedicated public and
    PyTree tests own more complex seeding behavior. Once any argument is
    already a ``LapTuple``, remaining plain arrays are constants.
    """
    if expected_fn is None:
        expected_fn = fn
    oracle_args = args
    if not any(isinstance(arg, LapTuple) for arg in args):
        oracle_args = tuple(make_laplacian_input(args))

    tracked_args = [arg for arg in oracle_args if isinstance(arg, LapTuple)]
    basis_size = tracked_args[0].dense_jacobian.shape[0]
    for arg in tracked_args:
        assert arg.dense_jacobian.shape[0] == basis_size

    primals = tuple(arg.x if isinstance(arg, LapTuple) else arg for arg in oracle_args)
    expected_x = expected_fn(*primals)

    y, jvp_fn = jax.linearize(expected_fn, *primals)

    jacobian_tangents = []
    laplacian_tangents = []
    flat_jacobian_blocks = []

    for arg, primal in zip(oracle_args, primals, strict=True):
        if isinstance(arg, LapTuple):
            dense_jacobian = arg.dense_jacobian
            jacobian_tangents.append(dense_jacobian)
            laplacian_tangents.append(arg.laplacian)
            flat_jacobian_blocks.append(dense_jacobian.reshape(basis_size, -1).T)
        else:
            zero_jacobian = jnp.zeros(
                (basis_size, *primal.shape),
                dtype=primal.dtype,
            )
            jacobian_tangents.append(zero_jacobian)
            laplacian_tangents.append(jnp.zeros_like(primal))
            flat_jacobian_blocks.append(zero_jacobian.reshape(basis_size, -1).T)

    expected_jacobian = jax.vmap(jvp_fn)(*jacobian_tangents)

    linear_laplacian = jvp_fn(*laplacian_tangents)

    flat_primals, unravel_primals = jax.flatten_util.ravel_pytree(primals)
    flat_y, unravel_output = jax.flatten_util.ravel_pytree(y)

    def flat_fn(flat):
        unflattened_primals = unravel_primals(flat)
        return jax.flatten_util.ravel_pytree(expected_fn(*unflattened_primals))[0]

    if jnp.iscomplexobj(flat_primals):
        if not jnp.iscomplexobj(flat_y):
            raise TypeError(
                "brute_force_oracle does not support real-valued outputs "
                "with complex-valued inputs; use a packed-real oracle or "
                "explicit contract assertions instead."
            )
        hessian = jax.hessian(flat_fn, holomorphic=True)(flat_primals)
    elif jnp.iscomplexobj(flat_y):
        hessian = jax.hessian(lambda flat: jnp.real(flat_fn(flat)))(flat_primals) + (
            1j * jax.hessian(lambda flat: jnp.imag(flat_fn(flat)))(flat_primals)
        )
    else:
        hessian = jax.hessian(flat_fn)(flat_primals)
    flat_input_jacobian = jnp.concatenate(flat_jacobian_blocks, axis=0)

    flat_hessian_laplacian = jnp.einsum(
        "oij,ik,jk->o",
        hessian,
        flat_input_jacobian,
        flat_input_jacobian,
    )
    hessian_laplacian = unravel_output(flat_hessian_laplacian)

    expected_laplacian = jax.tree.map(operator.add, linear_laplacian, hessian_laplacian)
    return LapTuple(expected_x, expected_jacobian, expected_laplacian)


def check_sparse_jacobian(
    fn: Callable,
    *args,
    expected_jacobian: type,
    actual_result: LapTuple | None = None,
    rtol=1e-5,
    atol=1e-10,
) -> LapTuple:
    """Run ``forward_laplacian``, assert sparse Jacobian type, and oracle-check."""
    if actual_result is None:
        actual_result = forward_laplacian(fn)(*args)
    assert isinstance(actual_result.jacobian, expected_jacobian)
    check_with_brute_force(fn, *args, actual_result=actual_result, rtol=rtol, atol=atol)
    return actual_result


def check_with_brute_force(
    fn: Callable,
    *args,
    expected_fn: Callable | None = None,
    actual_result: LapTuple | None = None,
    rtol=1e-5,
    atol=1e-10,
):
    """Check ``forward_laplacian(fn)(*args)`` against the dense chain rule.

    Plain-array calls cover simple top-level auto-seeding. Dedicated public and
    PyTree tests own more complex seeding behavior. Once any argument is
    already a ``LapTuple``, remaining plain arrays are constants.

    When ``actual_result`` is provided, compare that precomputed result against
    the oracle instead of rerunning ``forward_laplacian``.

    Current limitations (use direct assertions instead):
    - nested PyTree container inputs with selectively seeded leaves
    - PyTree-structured outputs from a single ``forward_laplacian`` call
    - batched ``vmap`` / ``shard_map`` layouts that only expose ``.laplacian``
    """
    if actual_result is None:
        actual_result = forward_laplacian(fn)(*args)
    expected = brute_force_oracle(fn, *args, expected_fn=expected_fn)
    assert_laptuple_allclose(actual_result, expected, rtol=rtol, atol=atol)
