# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse selection Jacobian behavior."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    forward_laplacian,
    make_laplacian_input,
)
from tests.laplacian.helpers import check_sparse_jacobian, check_with_brute_force
from tests.laplacian.input_fixtures import sparse_local1_input, sparse_local2_input

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(jnp.maximum, id="maximum"),
        pytest.param(jnp.minimum, id="minimum"),
    ),
)
def test_cross_axis_local1_selection_falls_back(op, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    lhs = sparse_local1_input(
        OwnerRole(0, np.array([0, 1], dtype=np.int32)),
        output_shape=(2, 2),
        input_shape=(2, 2),
    )
    rhs = sparse_local1_input(
        OwnerRole(1, np.array([0, 1], dtype=np.int32)),
        output_shape=(2, 2),
        input_shape=(2, 2),
    )
    actual = forward_laplacian(op)(lhs, rhs)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(op, lhs, rhs, actual_result=actual)
    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    assert record.levelno == logging.WARNING
    assert "not_implemented" in record.getMessage()


def test_mismatched_local1_maximum_falls_back(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    lhs = sparse_local1_input(
        OwnerRole(0, np.array([0, 1], dtype=np.int32)),
        output_shape=(2, 2),
        input_shape=(2, 2),
    )
    rhs = sparse_local1_input(
        OwnerRole(0, np.array([1, 0], dtype=np.int32)),
        output_shape=(2, 2),
        input_shape=(2, 2),
    )
    actual = forward_laplacian(jnp.maximum)(lhs, rhs)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(jnp.maximum, lhs, rhs, actual_result=actual)
    [record] = [
        record
        for record in caplog.records
        if "dense-fallback[max]" in record.getMessage()
    ]
    assert record.levelno == logging.WARNING
    assert "not_implemented" in record.getMessage()


@pytest.mark.parametrize(
    ("fn", "make_seed", "expected_jacobian"),
    (
        pytest.param(
            lambda value: jnp.maximum(value, 1.5),
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_plain_maximum",
        ),
        pytest.param(
            lambda value: jax.lax.select_n(
                jnp.array(0, dtype=jnp.int32),
                value,
                jnp.zeros((4, 4, 3), dtype=value.dtype),
            ),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_select_plain_branch",
        ),
        pytest.param(
            lambda value: jax.lax.select_n(
                jnp.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=jnp.int32),
                value,
                jnp.broadcast_to(value[:, :1] + 1.0, value.shape),
                jnp.broadcast_to(
                    jnp.full((1, 1), 7.0, dtype=value.dtype),
                    value.shape,
                ),
            ),
            lambda: make_laplacian_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_broadcast_mixed_branches",
        ),
    ),
)
def test_sparse_selection_operations(fn, make_seed, expected_jacobian):
    seed = make_seed()
    check_sparse_jacobian(fn, seed, expected_jacobian=expected_jacobian)
