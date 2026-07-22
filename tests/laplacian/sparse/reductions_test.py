# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse reduction Jacobian behavior."""

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
            lambda value: jnp.sum(value, axis=0),
            lambda: sparse_local1_input(
                OwnerRole(1, np.array([0, 1], dtype=np.int32)),
                output_shape=(4, 2, 3),
                input_shape=(2, 3),
                broadcast_blocks=True,
            ),
            Local1Jacobian,
            id="broadcast_local1_sum",
        ),
        pytest.param(
            lambda value: jnp.max(value, axis=0),
            lambda: sparse_local1_input(
                OwnerRole(1, np.array([0, 1], dtype=np.int32)),
                output_shape=(4, 2, 3),
                input_shape=(2, 3),
                broadcast_blocks=True,
            ),
            Local1Jacobian,
            id="broadcast_local1_max",
        ),
        pytest.param(
            lambda value: jnp.sum(value, axis=0),
            lambda: sparse_local1_input(
                OwnerRole(None, np.array([1], dtype=np.int32)),
            ),
            Local1Jacobian,
            id="constant_owner_sum",
        ),
        pytest.param(
            lambda value: jnp.max(value, axis=0),
            lambda: sparse_local1_input(
                OwnerRole(None, np.array([1], dtype=np.int32)),
            ),
            Local1Jacobian,
            id="constant_owner_max",
        ),
        pytest.param(
            lambda value: jnp.sum(value, axis=0),
            lambda: sparse_local1_input(
                OwnerRole(0, np.array([2, 2], dtype=np.int32)),
                output_shape=(2, 2),
                input_shape=(3, 2),
            ),
            Local1Jacobian,
            id="repeated_owner_sum",
        ),
        pytest.param(
            lambda value: jnp.max(value, axis=0),
            lambda: sparse_local1_input(
                OwnerRole(0, np.array([2, 2], dtype=np.int32)),
                output_shape=(2, 2),
                input_shape=(3, 2),
            ),
            Local1Jacobian,
            id="repeated_owner_max",
        ),
        pytest.param(
            lambda value: jnp.sum(value, axis=1),
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_non_owner_axis_sum",
        ),
        pytest.param(
            lambda value: jnp.sum(value, axis=2),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_feature_axis_sum",
        ),
        pytest.param(
            lambda value: jnp.max(value, axis=2),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_feature_axis_max",
        ),
    ),
)
def test_sparse_reductions(fn, make_seed, expected_jacobian):
    seed = make_seed()
    check_sparse_jacobian(fn, seed, expected_jacobian=expected_jacobian)
