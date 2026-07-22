# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Unary elementwise primitive semantics across all input cases."""

import jax
import jax.numpy as jnp
import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import (
    VECTOR_SHAPE,
    random_array,
    to_complex,
    tracked_case_input,
)

from .helpers import (
    check_binary,
    check_unary,
    parametrize_over_binary_cases,
    parametrize_over_tracked_cases,
)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(jnp.sin, "real", id="sin"),
        pytest.param(jnp.cos, "real", id="cos"),
        pytest.param(jnp.tan, "real", id="tan"),
        pytest.param(jnp.tanh, "real", id="tanh"),
        pytest.param(jnp.exp, "real", id="exp"),
        pytest.param(jnp.expm1, "real", id="expm1"),
        pytest.param(jnp.square, "real", id="square"),
        pytest.param(lambda v: v**3, "real", id="integer_pow"),
        pytest.param(jnp.arctan, "real", id="arctan"),
        pytest.param(jax.nn.sigmoid, "real", id="sigmoid"),
        pytest.param(jax.nn.softplus, "real", id="softplus"),
        pytest.param(jax.nn.silu, "real", id="silu"),
        pytest.param(jnp.abs, "real", id="abs"),
        pytest.param(jnp.round, "real", id="round"),
        pytest.param(jax.nn.relu, "real", id="relu"),
        pytest.param(jnp.log, "positive", id="log"),
        pytest.param(jnp.log1p, "positive", id="log1p"),
        pytest.param(jnp.sqrt, "positive", id="sqrt"),
        pytest.param(jax.lax.rsqrt, "positive", id="rsqrt"),
        pytest.param(jnp.arcsin, "unit", id="arcsin"),
        pytest.param(jnp.arccos, "unit", id="arccos"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_elementwise(case, op, domain):
    """The primitive in isolation: per-element derivative payloads."""
    check_unary(op, case, domain=domain)


@pytest.mark.parametrize(
    ("op", "domain"),
    [
        pytest.param(jnp.tanh, "real", id="tanh"),
        pytest.param(jnp.log, "positive", id="log"),
    ],
)
@parametrize_over_tracked_cases("case")
def test_elementwise_scalar_composition(case, op, domain):
    """Seeded state flowing through the op into a downstream reduction."""
    check_unary(lambda v: jnp.sum(op(v)), case, domain=domain)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(jnp.arctan2, "real", id="atan2"),
        pytest.param(jnp.logaddexp, "real", id="logaddexp"),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_binary_elementwise(lhs, rhs, op, domain):
    check_binary(lambda a, b: jnp.sum(op(a, b)), lhs, rhs, domain=domain)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(lambda a, b: jnp.arctan2(jnp.sum(a), b), "real", id="atan2_lhs"),
        pytest.param(lambda a, b: jnp.arctan2(a, jnp.sum(b)), "real", id="atan2_rhs"),
        pytest.param(
            lambda a, b: jnp.logaddexp(jnp.sum(a), b),
            "real",
            id="logaddexp_lhs",
        ),
        pytest.param(
            lambda a, b: jnp.logaddexp(a, jnp.sum(b)),
            "real",
            id="logaddexp_rhs",
        ),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_binary_elementwise_scalar_jacobian_broadcast(lhs, rhs, op, domain):
    check_binary(op, lhs, rhs, domain=domain, rtol=1e-4)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(
            lambda v: jnp.arctan2(v, random_array("unit", (3)) + 1.2),
            "real",
            id="atan2_const",
        ),
        pytest.param(
            lambda v: jnp.logaddexp(v, random_array("unit", (3))),
            "real",
            id="logaddexp_const",
        ),
        pytest.param(lambda v: v * jnp.sign(v), "real", id="sign_mul"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_constant_operand(case, op, domain):
    check_unary(op, case, domain=domain)


@parametrize_over_tracked_cases("case")
def test_sign_drops_derivatives(case):
    """jnp.sign output is piecewise constant: zero Jacobian and Laplacian."""
    check_unary(lambda v: v * 0.0 + jnp.sign(v), case)


@parametrize_over_tracked_cases("case")
def test_abs_at_zero(case):
    x = jnp.array(
        [
            [0.0, 1.0, -2.0],
            [0.5, 0.0, 1.5],
            [-1.0, 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    check_with_brute_force(jnp.abs, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_round_with_decimals(case):
    check_unary(lambda v: jnp.round(v, decimals=1), case)


@parametrize_over_tracked_cases("case")
def test_round_half_integer_to_nearest_even(case):
    x = jnp.array(
        [
            [0.5, 1.5, 2.5],
            [3.5, 4.5, 5.5],
            [6.5, 7.5, 8.5],
        ],
        dtype=jnp.float32,
    )
    check_with_brute_force(
        lambda value: jax.lax.round(
            value,
            rounding_method=jax.lax.RoundingMethod.TO_NEAREST_EVEN,
        ),
        tracked_case_input(x, case),
    )


@pytest.mark.parametrize(
    ("op", "offset"),
    (
        pytest.param(jnp.exp, 0j, id="complex_exp"),
        pytest.param(jnp.sin, 0j, id="complex_sin"),
        pytest.param(jnp.cos, 0j, id="complex_cos"),
        pytest.param(jnp.tan, 0j, id="complex_tan"),
        pytest.param(jnp.tanh, 0.5 + 0j, id="complex_tanh"),
        pytest.param(jnp.log, 2.0 + 0.5j, id="complex_log"),
        pytest.param(jnp.sqrt, 2.0 + 0.5j, id="complex_sqrt"),
        pytest.param(jnp.log1p, 2.0 + 0.5j, id="complex_log1p"),
        pytest.param(jnp.expm1, 0j, id="complex_expm1"),
        pytest.param(lambda z: z**3, 0j, id="complex_integer_pow"),
        pytest.param(lambda z: z**2.5, 2.0 + 0.5j, id="complex_fractional_pow"),
        pytest.param(jnp.abs, 2.0 + 0.5j, id="complex_abs"),
        pytest.param(lambda z: jnp.abs(z) ** 2, 0j, id="complex_abs_squared"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_complex_unary_scalar_output(case, op, offset):
    check_unary(
        lambda packed: jnp.sum(op(to_complex(packed) + offset)),
        case,
        shape=VECTOR_SHAPE,
    )


@parametrize_over_tracked_cases("case")
def test_complex_abs_vector_output(case):
    check_unary(
        lambda packed: jnp.abs(to_complex(packed) + (2.0 + 0.5j)),
        case,
        shape=VECTOR_SHAPE,
    )


@pytest.mark.parametrize(
    ("op", "lhs_offset", "rhs_offset"),
    (
        pytest.param(
            jnp.arctan2,
            0.0,
            1.0,
            id="atan2_quadrant_i",
        ),
        pytest.param(
            jnp.arctan2,
            -1.0,
            0.0,
            id="atan2_quadrant_ii",
        ),
        pytest.param(
            jnp.arctan2,
            0.0,
            -1.0,
            id="atan2_quadrant_iii",
        ),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_complex_atan2_quadrants(lhs, rhs, op, lhs_offset, rhs_offset):
    check_binary(
        lambda pa, pb: jnp.sum(
            op(
                to_complex(pa) + lhs_offset,
                to_complex(pb) + rhs_offset,
            )
        ),
        lhs,
        rhs,
        shape=VECTOR_SHAPE,
    )
