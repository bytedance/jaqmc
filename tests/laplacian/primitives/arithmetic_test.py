# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Arithmetic primitive semantics."""

import operator

import jax.numpy as jnp
import pytest

from tests.laplacian.input_fixtures import VECTOR_SHAPE, random_array, to_complex

from .helpers import (
    check_binary,
    check_unary,
    parametrize_over_binary_cases,
    parametrize_over_tracked_cases,
)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(operator.add, "real", id="add"),
        pytest.param(operator.sub, "real", id="sub"),
        pytest.param(operator.mul, "real", id="mul"),
        pytest.param(operator.truediv, "positive", id="div"),
        pytest.param(jnp.power, "positive", id="pow"),
        pytest.param(operator.mod, "real", id="rem"),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_binary_arithmetic(lhs, rhs, op, domain):
    check_binary(lambda a, b: jnp.sum(op(a, b)), lhs, rhs, domain=domain)


@parametrize_over_tracked_cases("case")
def test_neg(case):
    check_unary(jnp.negative, case)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        # One operand is a closed-over literal rather than a traced argument:
        # this exercises the jaxpr-constant handler path, distinct from the
        # plain-operand pairs in BINARY_PAIRS.
        pytest.param(
            lambda v: v + random_array("unit", (3)), "real", id="add_right_const"
        ),
        pytest.param(
            lambda v: random_array("unit", (3)) + v, "real", id="add_left_const"
        ),
        pytest.param(
            lambda v: v - random_array("unit", (3)), "real", id="sub_right_const"
        ),
        pytest.param(
            lambda v: random_array("unit", (3)) - v, "real", id="sub_left_const"
        ),
        pytest.param(
            lambda v: v * (random_array("unit", (3)) + 1.5), "real", id="mul_const"
        ),
        pytest.param(lambda v: v / 2.0, "real", id="div_scalar_const"),
        pytest.param(lambda v: 3.0 / v, "positive", id="rdiv_scalar_const"),
        pytest.param(lambda v: jnp.power(v, 2.5), "positive", id="pow_const_exponent"),
        pytest.param(lambda v: v % 1.5, "real", id="rem_scalar"),
        pytest.param(
            lambda v: v % (random_array("unit", (3)) + 1.5),
            "real",
            id="rem_const_divisor",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_constant_operand(case, op, domain):
    check_unary(op, case, domain=domain)


@parametrize_over_tracked_cases("case")
def test_constant_operand_rdiv_array_const(case):
    """An array constant numerator over a tracked denominator."""
    check_unary(
        lambda v: (random_array("unit", (3)) + 1.5) / v, case, domain="positive"
    )


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        # A scalar LapTuple factor broadcast against an array LapTuple: both
        # operands are tracked but with different ranks, exercising broadcast
        # merging.
        pytest.param(lambda v: jnp.sum(v) + v[None], "real", id="add"),
        pytest.param(lambda v: jnp.sum(v) - v[None], "real", id="sub"),
        pytest.param(lambda v: v[None] - jnp.sum(v), "real", id="rsub"),
        pytest.param(lambda v: jnp.sum(v) * v[None], "real", id="mul"),
        pytest.param(lambda v: jnp.sum(v) / v[None], "positive", id="div"),
        pytest.param(lambda v: v[None] / jnp.sum(v), "positive", id="rdiv"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_broadcast_scalar_factor(case, op, domain):
    check_unary(op, case, domain=domain, rtol=1e-4)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(lambda a, b: jnp.sum(a) + b, "real", id="scalar_lhs_add"),
        pytest.param(lambda a, b: a + jnp.sum(b), "real", id="scalar_rhs_add"),
        pytest.param(lambda a, b: jnp.sum(a) - b, "real", id="scalar_lhs_sub"),
        pytest.param(lambda a, b: a - jnp.sum(b), "real", id="scalar_rhs_sub"),
        pytest.param(lambda a, b: jnp.sum(a) * b, "real", id="scalar_lhs_mul"),
        pytest.param(lambda a, b: a * jnp.sum(b), "real", id="scalar_rhs_mul"),
        pytest.param(lambda a, b: jnp.sum(a) / b, "positive", id="scalar_lhs_div"),
        pytest.param(lambda a, b: a / jnp.sum(b), "positive", id="scalar_rhs_div"),
        pytest.param(lambda a, b: jnp.sum(a) ** b, "positive", id="scalar_lhs_pow"),
        pytest.param(lambda a, b: a ** jnp.sum(b), "positive", id="scalar_rhs_pow"),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_binary_scalar_jacobian_broadcasts_over_array_output(lhs, rhs, op, domain):
    """Scalar-output Jacobians broadcast correctly in two-input primitives."""
    check_binary(op, lhs, rhs, domain=domain, rtol=1e-4)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(lambda v: jnp.sum(v + v[:, :1]), "real", id="add_broadcast_col"),
        pytest.param(lambda v: jnp.sum(v * v[:1, :]), "real", id="mul_broadcast_row"),
        pytest.param(
            lambda v: jnp.sum(v / (v[:, :1] + 0.5)),
            "positive",
            id="div_broadcast_col",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_broadcast_row_column(case, op, domain):
    check_unary(op, case, domain=domain, rtol=1e-4)


@parametrize_over_tracked_cases("case")
def test_pow_plain_array_exponent_broadcast(case):
    exponents = random_array("unit", (3, 3), key=5) + 1.5
    check_unary(lambda v: jnp.sum(jnp.power(v, exponents)), case, domain="positive")


@pytest.mark.parametrize(
    ("op", "lhs_offset", "rhs_offset", "atol"),
    (
        pytest.param(operator.add, 0j, 0j, 1e-10, id="complex_add"),
        pytest.param(operator.sub, 0j, 0j, 1e-10, id="complex_sub"),
        pytest.param(operator.mul, 0j, 0j, 1e-10, id="complex_mul"),
        pytest.param(operator.truediv, 0j, 2.0 + 0.6j, 1e-10, id="complex_div"),
        pytest.param(operator.pow, 2.0 + 0.5j, 0j, 1e-4, id="complex_pow"),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_complex_binary_pairs(lhs, rhs, op, lhs_offset, rhs_offset, atol):
    check_binary(
        lambda pa, pb: jnp.sum(
            op(to_complex(pa) + lhs_offset, to_complex(pb) + rhs_offset)
        ),
        lhs,
        rhs,
        atol=atol,
        shape=VECTOR_SHAPE,
    )
