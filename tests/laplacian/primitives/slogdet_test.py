# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""slogdet primitive semantics across input cases.

The positive domain keeps every case's lifted matrices away from
singularity (pairwise-displacement matrices are exactly singular without the
domain offset: their self-pair row is zero).
"""

import jax.numpy as jnp
import pytest

from jaqmc.laplacian import LapTuple, forward_laplacian
from tests.laplacian.helpers import assert_allclose, check_with_brute_force
from tests.laplacian.input_fixtures import (
    MATRIX_SHAPE,
    random_array,
    to_complex,
    tracked_case_input,
)

from .helpers import check_unary, parametrize_over_tracked_cases

_COMPLEX_MATRIX_OFFSET = 1.0 + 0.5j


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(
            lambda v: jnp.sum(jnp.linalg.slogdet(v)[1]),
            "positive",
            id="logdet",
        ),
        pytest.param(
            lambda v: jnp.sum(jnp.linalg.slogdet(v)[1] ** 2),
            "positive",
            id="logdet_squared",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_logdet(case, op, domain):
    check_unary(op, case, domain=domain)


def test_plain_input_auto_seeds_slogdet_logdet_output():
    """Multi-output slogdet returns plain sign and tracked logdet for raw input."""
    x = random_array("positive")
    fl = forward_laplacian(jnp.linalg.slogdet)
    sign, logdet = fl(x)
    expected_sign, expected_logdet = jnp.linalg.slogdet(x)
    assert not isinstance(sign, LapTuple)
    assert_allclose(logdet.x, expected_logdet)
    assert_allclose(sign, expected_sign)


@parametrize_over_tracked_cases("case")
def test_real_sign_untracked_with_tracked_logdet(case):
    x = random_array("positive", shape=(3, 3), key=9)
    sign, logdet = forward_laplacian(jnp.linalg.slogdet)(tracked_case_input(x, case))
    assert not isinstance(sign, LapTuple)
    assert isinstance(logdet, LapTuple)
    expected_sign, expected_logdet = jnp.linalg.slogdet(x)
    assert_allclose(sign, expected_sign)
    assert_allclose(logdet.x, expected_logdet)


@parametrize_over_tracked_cases("case")
def test_batched_real_logdet(case):
    check_unary(
        lambda v: jnp.sum(jnp.linalg.slogdet(v.reshape(2, 2, 2))[1]),
        case,
        domain="positive",
        shape=(2, 2, 2),
    )


@parametrize_over_tracked_cases("case")
def test_batched_real_logdet_unreduced(case):
    x = random_array("positive", shape=(2, 2, 2), key=10)

    def fn(value):
        return jnp.linalg.slogdet(value.reshape(2, 2, 2))[1]

    check_with_brute_force(
        fn,
        tracked_case_input(x, case, key=11),
        rtol=1e-4,
        atol=1e-6,
    )


def test_composed_matrix_logdet_matches_brute_force():
    w = random_array("real", key=3, shape=(16, 16))

    def fn(v):
        _, logdet = jnp.linalg.slogdet(jnp.tanh((v @ w).reshape(4, 4)))
        return logdet

    check_with_brute_force(fn, random_array("real", key=4, shape=(16,)))


@parametrize_over_tracked_cases("case")
def test_complex_logdet(case):
    check_unary(
        lambda packed: jnp.sum(
            jnp.linalg.slogdet(to_complex(packed) + _COMPLEX_MATRIX_OFFSET)[1]
        ),
        case,
        shape=MATRIX_SHAPE,
    )


@parametrize_over_tracked_cases("case")
def test_complex_sign_real_part_derivatives(case):
    check_unary(
        lambda packed: jnp.sum(
            jnp.linalg.slogdet(to_complex(packed) + _COMPLEX_MATRIX_OFFSET)[0].real
        ),
        case,
        shape=MATRIX_SHAPE,
    )


@parametrize_over_tracked_cases("case")
def test_complex_sign_imag_part_derivatives(case):
    check_unary(
        lambda packed: jnp.sum(
            jnp.linalg.slogdet(to_complex(packed) + _COMPLEX_MATRIX_OFFSET)[0].imag
        ),
        case,
        shape=MATRIX_SHAPE,
    )


@parametrize_over_tracked_cases("case")
def test_batched_complex_sign(case):
    check_unary(
        lambda packed: jnp.sum(
            jnp.linalg.slogdet(to_complex(packed) + _COMPLEX_MATRIX_OFFSET)[0].real
        ),
        case,
        shape=(2, *MATRIX_SHAPE),
    )


@parametrize_over_tracked_cases("case")
def test_complex_sign_output(case):
    packed = random_array("real", shape=MATRIX_SHAPE, key=12)

    def fn(value):
        sign, _ = jnp.linalg.slogdet(to_complex(value) + _COMPLEX_MATRIX_OFFSET)
        return sign

    check_with_brute_force(fn, tracked_case_input(packed, case, key=13))


@parametrize_over_tracked_cases("case")
def test_batched_complex_sign_output(case):
    packed = random_array("real", shape=(2, *MATRIX_SHAPE), key=14)

    def fn(value):
        sign, _ = jnp.linalg.slogdet(to_complex(value) + _COMPLEX_MATRIX_OFFSET)
        return sign

    check_with_brute_force(
        fn,
        tracked_case_input(packed, case, key=15),
        rtol=1e-4,
        atol=1e-6,
    )
