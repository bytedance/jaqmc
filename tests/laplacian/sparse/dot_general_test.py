# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse dot_general Jacobian behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    forward_laplacian,
    make_laplacian_input,
)
from tests.laplacian.helpers import check_sparse_jacobian, check_with_brute_force
from tests.laplacian.input_fixtures import sparse_local1_input, sparse_local2_input


@pytest.mark.parametrize(
    ("fn", "make_seed", "expected_jacobian"),
    (
        pytest.param(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(10.0, dtype=jnp.float32).reshape(2, 5) / 10.0,
                dimension_numbers=(((0,), (0,)), ((), ())),
            ),
            lambda: make_laplacian_input(
                jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4),
                sparse_axis=1,
            ),
            Local1Jacobian,
            id="sparse_left_plain",
        ),
        pytest.param(
            lambda value: jax.lax.dot_general(
                jnp.arange(10.0, dtype=jnp.float32).reshape(5, 2) / 10.0,
                value,
                dimension_numbers=(((1,), (0,)), ((), ())),
            ),
            lambda: make_laplacian_input(
                jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4),
                sparse_axis=1,
            ),
            Local1Jacobian,
            id="sparse_right_plain",
        ),
        pytest.param(
            lambda value: (
                value @ (jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2) / 10.0)
            ),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_plain",
        ),
        pytest.param(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(40.0, dtype=jnp.float32).reshape(2, 4, 5),
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            ),
            lambda: make_laplacian_input(
                jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="batched_sparse_plain",
        ),
        pytest.param(
            lambda value: jax.lax.dot_general(
                jnp.broadcast_to(value, (4, *value.shape)),
                jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5),
                dimension_numbers=(((2,), (0,)), ((), ())),
            ),
            lambda: make_laplacian_input(
                jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="broadcast_sparse_plain",
        ),
        pytest.param(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                dimension_numbers=(((2,), (2,)), ((0, 1), (0, 1))),
            ),
            sparse_local2_input,
            Local2Jacobian,
            id="batched_local2_plain",
        ),
        pytest.param(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
                dimension_numbers=(((0,), (0,)), ((), ())),
            ),
            lambda: sparse_local1_input(
                OwnerRole(0, np.array([2, 2], dtype=np.int32)),
                output_shape=(2, 2),
                input_shape=(3, 2),
            ),
            Local1Jacobian,
            id="repeated_owner_local1",
        ),
    ),
)
def test_sparse_dot_general_operations(fn, make_seed, expected_jacobian):
    seed = make_seed()
    check_sparse_jacobian(fn, seed, expected_jacobian=expected_jacobian)


def test_batched_local1_dot_general():
    lhs = make_laplacian_input(
        jnp.arange(18.0, dtype=jnp.float32).reshape(2, 3, 3),
        sparse_axis=0,
    )
    rhs = make_laplacian_input(
        jnp.arange(18.0, dtype=jnp.float32).reshape(2, 3, 3)[::-1],
        sparse_axis=0,
    )
    fn = lambda left, right: jax.lax.dot_general(
        left,
        right,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )
    actual = forward_laplacian(fn)(lhs, rhs)
    assert isinstance(actual.jacobian, Local1Jacobian)
    check_with_brute_force(fn, lhs, rhs, actual_result=actual)


def test_two_local1_query_key_dot_promotes_to_local2():
    weight_q = jnp.arange(36.0, dtype=jnp.float32).reshape(6, 6) / 20.0
    weight_k = jnp.arange(36.0, dtype=jnp.float32).reshape(6, 6)[::-1] / 18.0

    def fn(value):
        query = jax.lax.reshape(
            jax.lax.dot_general(
                value,
                weight_q,
                dimension_numbers=(((1,), (0,)), ((), ())),
            ),
            new_sizes=(4, 2, 3),
            dimensions=None,
        )
        key = jax.lax.reshape(
            jax.lax.dot_general(
                value,
                weight_k,
                dimension_numbers=(((1,), (0,)), ((), ())),
            ),
            new_sizes=(4, 2, 3),
            dimensions=None,
        )
        return jax.lax.dot_general(
            query,
            key,
            dimension_numbers=(((2,), (2,)), ((1,), (1,))),
        )

    seed = make_laplacian_input(
        jnp.arange(24.0, dtype=jnp.float32).reshape(4, 6) / 10.0,
        sparse_axis=0,
    )
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, Local2Jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)
