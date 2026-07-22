# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse elementwise Jacobian behavior."""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    make_laplacian_input,
)
from tests.laplacian.helpers import check_sparse_jacobian
from tests.laplacian.input_fixtures import sparse_local1_input, sparse_local2_input


@pytest.mark.parametrize(
    ("fn", "make_seed", "expected_jacobian"),
    (
        pytest.param(
            jnp.exp,
            lambda: make_laplacian_input(
                jnp.linspace(-1.0, 1.0, 12).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_exp",
        ),
        pytest.param(
            jnp.exp,
            sparse_local2_input,
            Local2Jacobian,
            id="local2_exp",
        ),
        pytest.param(
            operator.neg,
            sparse_local2_input,
            Local2Jacobian,
            id="local2_neg",
        ),
        pytest.param(
            lambda value: jax.lax.round(
                value,
                rounding_method=jax.lax.RoundingMethod.AWAY_FROM_ZERO,
            ),
            lambda: sparse_local1_input(
                OwnerRole(1, np.array([0, 1], dtype=np.int32)),
                output_shape=(4, 2, 3),
                input_shape=(2, 3),
                broadcast_blocks=True,
            ),
            Local1Jacobian,
            id="broadcast_local1_round",
        ),
    ),
)
def test_sparse_elementwise_operations(fn, make_seed, expected_jacobian):
    seed = make_seed()
    check_sparse_jacobian(fn, seed, expected_jacobian=expected_jacobian)
