# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse dtype-conversion Jacobian behavior."""

import jax
import jax.numpy as jnp
import pytest

from jaqmc.laplacian import Local1Jacobian, Local2Jacobian, make_laplacian_input
from tests.laplacian.helpers import check_sparse_jacobian
from tests.laplacian.input_fixtures import sparse_local2_input


@pytest.mark.parametrize(
    ("fn", "make_seed", "expected_jacobian"),
    (
        pytest.param(
            lambda value: jax.lax.convert_element_type(value, jnp.float32),
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float16).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_float_cast",
        ),
        pytest.param(
            lambda value: jax.lax.convert_element_type(value, jnp.complex64),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_complex_cast",
        ),
    ),
)
def test_sparse_dtype_conversions(fn, make_seed, expected_jacobian):
    seed = make_seed()
    check_sparse_jacobian(fn, seed, expected_jacobian=expected_jacobian)
