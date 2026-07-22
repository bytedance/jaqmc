# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse shape and concatenation Jacobian behavior."""

import logging
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    LapTuple,
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    forward_laplacian,
    make_laplacian_input,
)
from tests.laplacian.helpers import check_sparse_jacobian, check_with_brute_force
from tests.laplacian.input_fixtures import (
    make_local1_input,
    make_local2_input,
    sparse_local1_input,
    sparse_local2_input,
)

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


def _varying_owner_local1(owner: OwnerRole) -> LapTuple:
    x = jnp.arange(8.0, dtype=jnp.float32).reshape(2, 2, 2)
    return make_local1_input(
        x,
        blocks=jnp.arange(8.0, dtype=jnp.float32).reshape(1, 1, *x.shape),
        owners=OwnerRoles(owner),
        input_shape=(2, 1),
    )


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(lambda value: jnp.transpose(value, (2, 0, 1)), id="transpose"),
        pytest.param(lambda value: jnp.squeeze(value, axis=1), id="squeeze"),
        pytest.param(lambda value: jnp.flip(value, axis=0), id="reverse"),
        pytest.param(
            operator.itemgetter((slice(None), slice(None), slice(1, None))),
            id="slice",
        ),
        pytest.param(
            lambda value: jnp.broadcast_to(value, (4, *value.shape)),
            id="broadcast",
        ),
        pytest.param(
            lambda value: value * jnp.ones((4, *value.shape), dtype=value.dtype),
            id="sparse_leading_broadcast",
        ),
        pytest.param(lambda value: jnp.reshape(value, (1, 2, 3)), id="reshape"),
    ),
)
def test_constant_owner_local1_shape_operations(fn):
    seed = sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32)))
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, Local1Jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(lambda value: jnp.transpose(value, (1, 0, 2)), id="transpose"),
        pytest.param(lambda value: jnp.reshape(value, (4, 4, 1, 3)), id="reshape"),
        pytest.param(
            lambda value: jnp.broadcast_to(value, (2, *value.shape)),
            id="broadcast",
        ),
        pytest.param(lambda value: jnp.flip(value, axis=0), id="reverse"),
        pytest.param(
            operator.itemgetter((slice(None), slice(None), slice(1, None))),
            id="slice",
        ),
        pytest.param(
            lambda value: jnp.concatenate([value[:, :2], value[:, 2:]], axis=1),
            id="concatenate_owner_axis",
        ),
        pytest.param(
            lambda value: jnp.concatenate(
                [jnp.zeros((4, 4, 2), dtype=value.dtype), value],
                axis=2,
            ),
            id="concatenate_plain_segment",
        ),
    ),
)
def test_local2_shape_operations(fn):
    seed = sparse_local2_input()
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, Local2Jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)


@pytest.mark.parametrize(
    ("fn", "make_args", "expected_jacobian"),
    (
        pytest.param(
            lambda left, right: jnp.concatenate([left, right], axis=0),
            lambda: (
                sparse_local1_input(OwnerRole(None, np.array([0], dtype=np.int32))),
                sparse_local1_input(OwnerRole(None, np.array([2], dtype=np.int32))),
            ),
            Local1Jacobian,
            id="constant_owner_output_axis",
        ),
        pytest.param(
            lambda left, right: jnp.concatenate([left, right], axis=2),
            lambda: (
                sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32))),
                sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32))),
            ),
            Local1Jacobian,
            id="matching_constant_owner_off_axis",
        ),
        pytest.param(
            lambda left, right: jnp.concatenate([left, right], axis=2),
            lambda: (
                sparse_local1_input(OwnerRole(None, np.array([0], dtype=np.int32))),
                sparse_local1_input(OwnerRole(None, np.array([2], dtype=np.int32))),
            ),
            Local1Jacobian,
            id="distinct_constant_owner_off_axis",
        ),
        pytest.param(
            lambda left, right, other: jnp.concatenate([left, right], axis=2) * other,
            lambda: (
                sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32))),
                sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32))),
                sparse_local1_input(
                    OwnerRole(None, np.array([1], dtype=np.int32)),
                    output_shape=(2, 1, 6),
                    input_shape=(3, 1),
                ),
            ),
            Local1Jacobian,
            id="matching_constant_owner_composition",
        ),
    ),
)
def test_constant_owner_concatenation(fn, make_args, expected_jacobian):
    args = make_args()
    check_sparse_jacobian(fn, *args, expected_jacobian=expected_jacobian)


@pytest.mark.parametrize(
    ("lhs_owner", "rhs_owner"),
    (
        pytest.param(
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(0, np.array([1, 0], dtype=np.int32)),
            id="different_off_axis_owners",
        ),
        pytest.param(
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(1, np.array([1, 0], dtype=np.int32)),
            id="mixed_off_axis_owner_axes",
        ),
    ),
)
def test_off_axis_owner_concatenation_falls_back(lhs_owner, rhs_owner, caplog):
    """Concatenation densifies when off-axis owner layouts cannot be merged."""
    lhs = _varying_owner_local1(lhs_owner)
    rhs = _varying_owner_local1(rhs_owner)
    fn = lambda left, right: jnp.concatenate([left, right], axis=2)

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(fn)(lhs, rhs)

    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(fn, lhs, rhs, actual_result=actual)
    [record] = [
        record
        for record in caplog.records
        if "dense-fallback[concatenate]" in record.getMessage()
    ]
    assert record.levelno == logging.DEBUG
    assert "unrepresentable" in record.getMessage()


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(operator.itemgetter(slice(1, None)), id="slice"),
        pytest.param(
            lambda value: jnp.concatenate([value, value], axis=0),
            id="concatenate",
        ),
    ),
)
def test_broadcast_filled_local1_shape_operations(fn):
    seed = sparse_local1_input(
        OwnerRole(1, np.array([0, 1], dtype=np.int32)),
        output_shape=(4, 2, 3),
        input_shape=(2, 3),
        broadcast_blocks=True,
    )
    check_sparse_jacobian(fn, seed, expected_jacobian=Local1Jacobian)


@pytest.mark.parametrize(
    "fn",
    (
        pytest.param(
            lambda value: jnp.concatenate(
                [jnp.broadcast_to(value, (2, *value.shape))] * 2,
                axis=0,
            ),
            id="concatenate_after_broadcast",
        ),
    ),
)
def test_identity_local1_broadcast_shape_compositions(fn):
    seed = make_laplacian_input(
        jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
        sparse_axis=0,
    )
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, Local1Jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)


def test_identity_local1_broadcast_flattening_falls_back_to_dense():
    seed = make_laplacian_input(
        jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
        sparse_axis=0,
    )
    fn = lambda value: jnp.reshape(jnp.broadcast_to(value, (2, *value.shape)), (2, 6))
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(fn, seed, actual_result=actual)


@pytest.mark.parametrize(
    ("fn", "expected_level", "expected_kind"),
    (
        pytest.param(
            lambda value: jax.lax.reshape(value, new_sizes=(3, 2), dimensions=(1, 0)),
            logging.WARNING,
            "not_implemented",
            id="permutation",
        ),
        pytest.param(
            lambda value: jnp.reshape(value, (6,)),
            logging.DEBUG,
            "unrepresentable",
            id="flattening",
        ),
    ),
)
def test_sparse_reshape_fallback_policy(fn, expected_level, expected_kind, caplog):
    seed = make_laplacian_input(
        jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
        sparse_axis=0,
    )
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(fn, seed, actual_result=actual)
    [record] = [
        record
        for record in caplog.records
        if "dense-fallback[reshape]" in record.getMessage()
    ]
    assert record.levelno == expected_level
    assert expected_kind in record.getMessage()


def test_dense_and_local1_concatenation():
    x = jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3)
    lhs = make_laplacian_input(x[:2])
    rhs = make_laplacian_input(x[2:], sparse_axis=0)
    fn = lambda left, right: jnp.concatenate([left, right], axis=0)
    actual = forward_laplacian(fn)(lhs, rhs)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(fn, lhs, rhs, actual_result=actual)


def test_local1_and_local2_concatenation():
    x = jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3)
    lhs = make_laplacian_input(x[:2], sparse_axis=0)
    rhs = make_local2_input(
        x[2:],
        blocks=jnp.arange(36.0, dtype=jnp.float32).reshape(2, 3, 2, 3),
        owners=OwnerRoles(
            OwnerRole(0, np.arange(2, dtype=np.int32)),
            OwnerRole(0, np.arange(2, dtype=np.int32)),
        ),
        input_shape=(2, 3),
    )
    fn = lambda left, right: jnp.concatenate([left, right], axis=0)
    actual = forward_laplacian(fn)(lhs, rhs)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(fn, lhs, rhs, actual_result=actual)
