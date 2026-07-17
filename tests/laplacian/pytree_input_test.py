# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""PyTree and multi-argument public input behavior for Forward Laplacian."""

import jax
import jax.numpy as jnp

from jaqmc.laplacian import forward_laplacian, make_laplacian_input
from tests.laplacian.helpers import assert_allclose, check_with_brute_force


def test_nested_pytree_input_and_output():
    """Nested pytrees preserve output structure and share one tracked basis."""
    a = jnp.array([0.5, -0.3], dtype=jnp.float32)
    b = jnp.array([0.8, 0.4], dtype=jnp.float32)
    x = {"a": a, "b": (b,)}

    def fn(tree):
        lhs = tree["a"]
        rhs = tree["b"][0]
        return {
            "product": lhs * rhs,
            "summary": (jnp.sum(lhs**2) + jnp.sum(jnp.sin(rhs)),),
        }

    result = forward_laplacian(fn)(x)

    assert set(result) == {"product", "summary"}
    assert_allclose(result["product"].x, a * b)
    expected_product_jac = jax.jacfwd(lambda tree: fn(tree)["product"])(x)
    expected_product_jac = jnp.concatenate(
        [
            expected_product_jac["a"].reshape(2, -1),
            expected_product_jac["b"][0].reshape(2, -1),
        ],
        axis=-1,
    )
    expected_product_jac = jnp.moveaxis(expected_product_jac, -1, 0)
    assert_allclose(result["product"].dense_jacobian, expected_product_jac)
    assert_allclose(result["summary"][0].x, fn(x)["summary"][0])
    expected_summary_jac = jax.jacfwd(lambda tree: fn(tree)["summary"][0])(x)
    expected_summary_jac = jnp.concatenate(
        [
            expected_summary_jac["a"].reshape(-1),
            expected_summary_jac["b"][0].reshape(-1),
        ]
    )
    assert_allclose(result["summary"][0].dense_jacobian, expected_summary_jac)
    assert_allclose(result["product"].laplacian, jnp.zeros_like(a * b))
    assert_allclose(
        result["summary"][0].laplacian,
        2 * a.size - jnp.sum(jnp.sin(b)),
    )


def test_nested_wrapped_leaf_selects_only_that_leaf():
    """A LapTuple leaf makes sibling plain leaves constants."""
    x = jnp.array([0.5, -0.3], dtype=jnp.float32)
    scale = jnp.array([0.8, 0.4], dtype=jnp.float32)

    def fn(tree):
        return jnp.sum(tree["x"] ** 2 * tree["scale"])

    result = forward_laplacian(fn)({"x": make_laplacian_input(x), "scale": scale})

    assert_allclose(result.x, jnp.sum(x**2 * scale))
    assert result.dense_jacobian.shape == (2,)
    assert_allclose(result.dense_jacobian, 2 * x * scale)
    assert_allclose(result.laplacian, jnp.sum(2 * scale))


def test_two_plain_positional_args_auto_seed_separate_broadcast_basis_blocks():
    """Auto-seeded multi-arg calls concatenate each argument's basis block."""
    a = jnp.array([[0.7]])
    b = jnp.ones((1, 1, 16))

    def fn(lhs, rhs):
        return lhs[:, None, :, None] * rhs

    result = forward_laplacian(fn)(a, b)

    assert_allclose(result.dense_jacobian[: a.size], b.reshape(1, 1, 1, 1, 16))
    expected_b_block = (
        jnp.eye(b.size, dtype=b.dtype).reshape(b.size, 1, *b.shape)
        * a[:, None, :, None]
    )
    assert_allclose(result.dense_jacobian[a.size :], expected_b_block)
    check_with_brute_force(fn, a, b, actual_result=result, rtol=1e-4)
