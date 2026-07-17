# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Complex component primitive semantics."""

import jax.numpy as jnp
import pytest

from jaqmc.laplacian import LapTuple, forward_laplacian
from tests.laplacian.helpers import assert_allclose
from tests.laplacian.input_fixtures import (
    VECTOR_SHAPE,
    random_array,
    to_complex,
    tracked_case_input,
)

from .helpers import check_unary, parametrize_over_tracked_cases


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(jnp.conj, id="conj"),
        pytest.param(jnp.real, id="real"),
        pytest.param(jnp.imag, id="imag"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_complex_componentwise_vector_output(case, op):
    check_unary(
        lambda packed: op(to_complex(packed)),
        case,
        shape=VECTOR_SHAPE,
    )


@parametrize_over_tracked_cases("case")
def test_complex_conj_squared(case):
    check_unary(
        lambda packed: jnp.sum(jnp.conj(to_complex(packed)) ** 2),
        case,
        shape=VECTOR_SHAPE,
    )


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(lambda z: jnp.real(z) ** 2, id="real_squared"),
        pytest.param(lambda z: jnp.imag(z) ** 2, id="imag_squared"),
        pytest.param(lambda z: jnp.real(jnp.sin(z)), id="sin_real"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_complex_projection_compositions(case, op):
    check_unary(
        lambda packed: jnp.sum(op(to_complex(packed))),
        case,
        shape=VECTOR_SHAPE,
    )


@parametrize_over_tracked_cases("case")
def test_imag_of_real_input_preserves_zero_derivative_state(case):
    """Imag on a zero-imag complex value returns a tracked zero state."""
    fl = forward_laplacian(lambda value: jnp.imag(value.astype(jnp.complex64)))
    x = random_array("real")
    result = fl(tracked_case_input(x, case))
    assert isinstance(result, LapTuple)
    assert_allclose(result.x, jnp.zeros_like(x))
    assert_allclose(result.dense_jacobian, jnp.zeros_like(result.dense_jacobian))
    assert_allclose(result.laplacian, jnp.zeros_like(x))
