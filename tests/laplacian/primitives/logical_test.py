# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Logical, equality, and predicate primitive semantics."""

import jax.numpy as jnp
import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import tracked_case_input

from .helpers import (
    check_binary,
    check_unary,
    parametrize_over_binary_cases,
    parametrize_over_tracked_cases,
)

_EQUALITY_X = jnp.array(
    [
        [0.1, 0.1, 0.2],
        [0.1, -0.3, 0.4],
        [0.5, 0.1, 0.6],
    ],
    dtype=jnp.float32,
)

_EQUALITY_BRANCH_X = jnp.array(
    [
        [0.1, 0.2, 0.3],
        [0.1, 0.4, 0.5],
        [0.6, 0.7, 0.8],
    ],
    dtype=jnp.float32,
)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(lambda v: jnp.where(v > 0.1, v**2, v), "real", id="gt"),
        pytest.param(lambda v: jnp.where(v >= 0.1, v**2, v), "real", id="ge"),
        pytest.param(lambda v: jnp.where(v < 0.1, v**2, v), "real", id="lt"),
        pytest.param(lambda v: jnp.where(v <= 0.1, v**2, v), "real", id="le"),
        pytest.param(
            lambda v: jnp.where(jnp.logical_and(v > -0.5, v < 0.5), v**2, v),
            "real",
            id="logical_and",
        ),
        pytest.param(
            lambda v: jnp.where(jnp.logical_or(v < -0.5, v > 0.5), v**2, v),
            "real",
            id="logical_or",
        ),
        pytest.param(
            lambda v: jnp.where(jnp.logical_xor(v > -0.5, v > 0.5), v**2, v),
            "real",
            id="logical_xor",
        ),
        pytest.param(
            lambda v: jnp.where(jnp.logical_not(v > 0), v, v**3),
            "real",
            id="logical_not",
        ),
        pytest.param(
            lambda v: jnp.where(jnp.isfinite(v), v**2, v), "real", id="isfinite"
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_logical(case, op, domain):
    check_unary(op, case, domain=domain)


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(
            lambda v: jnp.where(jnp.equal(v, 0.1), v**2, v),
            id="eq",
        ),
        pytest.param(
            lambda v: jnp.where(jnp.not_equal(v, 0.1), v**2, v),
            id="ne",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_logical_equality_branches(case, op):
    check_with_brute_force(op, tracked_case_input(_EQUALITY_BRANCH_X, case))


@pytest.mark.parametrize(
    ("op", "x"),
    (
        pytest.param(
            lambda v: jnp.where(jnp.isclose(v, 0.1), v**2, v),
            _EQUALITY_X,
            id="eq_mixed_branches",
        ),
        pytest.param(
            lambda v: jnp.where(~jnp.isclose(v, 0.1), v**2, v),
            _EQUALITY_X,
            id="ne_mixed_branches",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_logical_equality_mixed_branches(case, op, x):
    check_with_brute_force(op, tracked_case_input(x, case))


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(a > b, a**2, b)),
            id="gt_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(a >= b, a**2, b)),
            id="ge_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(a < b, a**2, b)),
            id="lt_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(a <= b, a**2, b)),
            id="le_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(a == b, a**2, b)),
            id="eq_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(a != b, a**2, b)),
            id="ne_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(
                jnp.where(jnp.logical_and(a > -0.5, b < 0.5), a**2, b)
            ),
            id="logical_and_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(jnp.logical_or(a < -0.5, b > 0.5), a**2, b)),
            id="logical_or_two_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.where(jnp.logical_xor(a > 0.0, b > 0.0), a**2, b)),
            id="logical_xor_two_tracked",
        ),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_logical_binary_predicates(lhs, rhs, op):
    check_binary(op, lhs, rhs)


@parametrize_over_tracked_cases("case")
def test_logical_nonfinite_predicates(case):
    x = jnp.array(
        [
            [0.0, jnp.inf, -jnp.inf],
            [jnp.nan, 1.0, -1.0],
            [2.0, -2.0, 0.5],
        ],
        dtype=jnp.float32,
    )
    check_with_brute_force(
        lambda v: jnp.where(jnp.isfinite(v), v, jnp.zeros_like(v)),
        tracked_case_input(x, case),
    )
