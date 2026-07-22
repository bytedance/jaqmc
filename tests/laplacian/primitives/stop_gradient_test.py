# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""stop_gradient primitive semantics."""

import jax
import jax.numpy as jnp
import pytest

from jaqmc.laplacian import LapTuple, forward_laplacian
from tests.laplacian.helpers import assert_allclose
from tests.laplacian.input_fixtures import tracked_case_input

from .helpers import check_unary, parametrize_over_tracked_cases


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(
            lambda v: v * jax.lax.stop_gradient(v),
            "real",
            id="stop_gradient",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_stop_gradient(case, op, domain):
    check_unary(op, case, domain=domain)


@parametrize_over_tracked_cases("case")
def test_stop_gradient_drops_derivative_payload(case):
    x = tracked_case_input(
        jnp.array(
            [
                [0.5, -0.3, 1.2],
                [0.8, -0.1, 0.4],
                [0.2, 0.6, -0.7],
            ],
            dtype=jnp.float32,
        ),
        case,
    )
    assert isinstance(x, LapTuple)
    result = forward_laplacian(jax.lax.stop_gradient)(x)
    assert not isinstance(result, LapTuple)
    assert_allclose(result, x.x)
