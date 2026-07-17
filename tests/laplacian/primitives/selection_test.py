# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Selection primitive semantics.

``where`` coverage here exercises the lowered ``select_n`` path; max and min
have their own pointwise selection handlers.
"""

import jax
import jax.numpy as jnp
import pytest

from jaqmc.laplacian import LapTuple, forward_laplacian, make_laplacian_input
from tests.laplacian.helpers import assert_allclose, check_with_brute_force
from tests.laplacian.input_fixtures import random_array, tracked_case_input

from .helpers import (
    check_binary,
    parametrize_over_binary_cases,
    parametrize_over_tracked_cases,
)


@parametrize_over_tracked_cases("case")
def test_select_n_ignores_tracked_selector_derivatives(case):
    primal_selector = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    selector_seed = tracked_case_input(
        jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
        case,
        input_shape=(2, 5),
    )
    assert isinstance(selector_seed, LapTuple)
    selector = LapTuple(
        primal_selector,
        jnp.ones_like(selector_seed.dense_jacobian),
        jnp.ones_like(selector_seed.laplacian),
    )
    data = tracked_case_input(
        jnp.array([[10.0, 11.0], [20.0, 21.0]], dtype=jnp.float32),
        case,
        input_shape=(2, 5),
    )

    def fn(tracked_selector, tracked_data):
        return jax.lax.select_n(
            tracked_selector,
            jnp.zeros((2, 2), dtype=jnp.float32),
            tracked_data,
        )

    out = forward_laplacian(fn)(selector, data)
    expected = forward_laplacian(fn)(primal_selector, data)
    assert_allclose(out.x, expected.x)
    assert_allclose(out.dense_jacobian, expected.dense_jacobian)
    assert_allclose(out.laplacian, expected.laplacian)


@parametrize_over_tracked_cases("case")
def test_select_n_tracked_and_plain_branches(case):
    x = random_array(shape=(3, 3))

    def fn(value):
        return jax.lax.select_n(
            jnp.array(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0]],
                dtype=jnp.int32,
            ),
            value,
            value + jnp.asarray(1.0, dtype=value.dtype),
            jnp.full_like(value, 7.0),
        )

    check_with_brute_force(fn, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_select_n_all_tracked_branches(case):
    x = random_array(shape=(3, 3))

    def fn(value):
        return jax.lax.select_n(
            jnp.array(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0]],
                dtype=jnp.int32,
            ),
            value,
            value + jnp.asarray(1.0, dtype=value.dtype),
            value * jnp.asarray(2.0, dtype=value.dtype),
        )

    check_with_brute_force(fn, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_select_n_clips_out_of_range_selector(case):
    x = random_array(shape=(3, 3))

    def fn(value):
        return jax.lax.select_n(
            jnp.array(
                [[-1, 0, 99], [2, 1, 0], [0, 99, -5]],
                dtype=jnp.int32,
            ),
            value,
            value + jnp.asarray(1.0, dtype=value.dtype),
            jnp.full_like(value, 7.0),
        )

    check_with_brute_force(fn, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_select_n_scalar_selector_broadcast(case):
    x = random_array(shape=(3, 3))

    def fn(value):
        return jax.lax.select_n(
            jnp.array(1, dtype=jnp.int32),
            value,
            value + jnp.asarray(1.0, dtype=value.dtype),
            value * jnp.asarray(2.0, dtype=value.dtype),
        )

    check_with_brute_force(fn, tracked_case_input(x, case))


def test_select_n_all_plain_branches_drops_tracking():
    x = random_array(shape=(3, 3))
    selector = jnp.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=jnp.int32)
    plain_cases = (
        jnp.full_like(x, 1.0),
        jnp.full_like(x, 2.0),
        jnp.full_like(x, 3.0),
    )

    def fn(_value):
        return jax.lax.select_n(selector, *plain_cases)

    result = forward_laplacian(fn)(make_laplacian_input(x))
    expected = jax.lax.select_n(selector, *plain_cases)
    assert not isinstance(result, LapTuple)
    assert_allclose(result, expected)


@parametrize_over_tracked_cases("case")
def test_select_n_broadcasted_mixed_branches(case):
    x = random_array(shape=(3, 3, 2))

    def fn(value):
        return jax.lax.select_n(
            jnp.array(2, dtype=jnp.int32),
            value,
            jnp.broadcast_to(value[:, :, :1] + 1.0, value.shape),
            jnp.broadcast_to(
                jnp.full((1, 1, 1), 7.0, dtype=value.dtype),
                value.shape,
            ),
        )

    check_with_brute_force(fn, tracked_case_input(x, case))


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(jnp.maximum, "real", id="maximum"),
        pytest.param(jnp.minimum, "real", id="minimum"),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_binary_selection(lhs, rhs, op, domain):
    check_binary(lambda a, b: jnp.sum(op(a, b)), lhs, rhs, domain=domain)


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(lambda a, b: jnp.maximum(jnp.sum(a), b), id="maximum_lhs"),
        pytest.param(lambda a, b: jnp.maximum(a, jnp.sum(b)), id="maximum_rhs"),
        pytest.param(lambda a, b: jnp.minimum(jnp.sum(a), b), id="minimum_lhs"),
        pytest.param(lambda a, b: jnp.minimum(a, jnp.sum(b)), id="minimum_rhs"),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_binary_selection_scalar_jacobian_broadcast(lhs, rhs, op):
    check_binary(op, lhs, rhs, rtol=1e-4)


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(
            lambda a, b: jnp.where(a > b, jnp.sum(a), b),
            id="where_scalar_lhs_branch",
        ),
        pytest.param(
            lambda a, b: jnp.where(a > b, a, jnp.sum(b)),
            id="where_scalar_rhs_branch",
        ),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_where_lowered_select_n_scalar_branch_jacobian_broadcast(lhs, rhs, op):
    check_binary(op, lhs, rhs, rtol=1e-4)


@pytest.mark.parametrize(
    ("op", "lhs", "rhs"),
    (
        pytest.param(
            jnp.maximum,
            jnp.array(
                [
                    [2.0, 2.0, 1.0],
                    [0.0, 3.0, 3.0],
                    [4.0, 1.0, 4.0],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [0.0, 3.0, 2.0],
                    [4.0, 1.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
            id="maximum_ties_lhs_wins",
        ),
        pytest.param(
            jnp.minimum,
            jnp.array(
                [
                    [-2.0, -2.0, 1.0],
                    [0.0, -3.0, -3.0],
                    [-4.0, 1.0, -4.0],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [-1.0, -2.0, 2.0],
                    [0.0, -3.0, -2.0],
                    [-4.0, 2.0, -3.0],
                ],
                dtype=jnp.float32,
            ),
            id="minimum_ties_lhs_wins",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_pointwise_selection_ties(case, op, lhs, rhs):
    check_with_brute_force(
        op,
        tracked_case_input(lhs, case, key=2, input_shape=(3, 5)),
        tracked_case_input(rhs, case, key=3, input_shape=(3, 5)),
    )


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(
            lambda v: jnp.sum(jnp.maximum(v, v[:, :1])),
            id="maximum_broadcast_col",
        ),
        pytest.param(
            lambda v: jnp.sum(jnp.minimum(v, v[:1, :])),
            id="minimum_broadcast_row",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_pointwise_selection_broadcast(case, fn):
    x = random_array(shape=(3, 3, 2))
    check_with_brute_force(fn, tracked_case_input(x, case))


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(
            lambda v: jnp.sum(jnp.maximum(v, 1.5)),
            id="maximum_literal_scalar",
        ),
        pytest.param(
            lambda v: jnp.sum(jnp.minimum(v, -0.5)),
            id="minimum_literal_scalar",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_pointwise_selection_literal_plain_operands(case, fn):
    x = random_array(shape=(3, 3))
    check_with_brute_force(fn, tracked_case_input(x, case))


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(
            lambda v: jnp.sum(jnp.maximum(v, v[:, :1] + 0.25)),
            id="maximum_derived_broadcast_col",
        ),
        pytest.param(
            lambda v: jnp.sum(jnp.minimum(v, v[:1, :] - 0.25)),
            id="minimum_derived_broadcast_row",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_pointwise_selection_derived_broadcast_operands(case, fn):
    x = random_array(shape=(3, 3, 2))
    check_with_brute_force(fn, tracked_case_input(x, case))
