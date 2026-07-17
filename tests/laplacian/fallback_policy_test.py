# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Dense fallback kind and visibility policy for Forward Laplacian."""

import logging
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaqmc.laplacian.hessian as laplacian_hessian
from jaqmc.laplacian import (
    LapTuple,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    forward_laplacian,
    make_laplacian_input,
)
from jaqmc.laplacian.primitives.core import log_dense_fallback
from tests.laplacian.sparse.helpers import mismatched_local1_pair

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


def test_linear_complex_avoids_generic_hessian(monkeypatch):
    """The registered linear complex rule must not use generic fallback."""
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    seed = LapTuple(x, jnp.ones((3, *x.shape)), jnp.zeros_like(x))

    def fail_if_hessian_is_materialized(*args, **kwargs):
        raise AssertionError("linear complex primitive materialized a Hessian")

    monkeypatch.setattr(
        laplacian_hessian,
        "JHJ_via_hessian",
        fail_if_hessian_is_materialized,
    )
    result = forward_laplacian(
        lambda real: jax.lax.complex(real, jnp.zeros_like(real))
    )(seed)

    np.testing.assert_allclose(result.x, jax.lax.complex(x, jnp.zeros_like(x)))
    np.testing.assert_allclose(
        result.dense_jacobian,
        seed.dense_jacobian.astype(jnp.complex64),
    )
    np.testing.assert_allclose(
        result.laplacian,
        jnp.zeros_like(x, dtype=jnp.complex64),
    )


def test_not_implemented_fallback_logs_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    log_dense_fallback(
        site="policy-test",
        kind="not_implemented",
        reason="free-form reason is not part of this contract",
    )

    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    message = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "dense-fallback[policy-test]" in message
    assert "not_implemented" in message


def test_unrepresentable_fallback_logs_debug(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    log_dense_fallback(
        site="policy-test",
        kind="unrepresentable",
        reason="free-form reason is not part of this contract",
    )

    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    message = record.getMessage()
    assert record.levelno == logging.DEBUG
    assert "dense-fallback[policy-test]" in message
    assert "unrepresentable" in message


def test_sparse_reshape_permutation_logs_not_implemented(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    seed = make_laplacian_input(
        jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
        sparse_axis=0,
    )

    forward_laplacian(
        lambda value: jax.lax.reshape(value, new_sizes=(3, 2), dimensions=(1, 0))
    )(seed)

    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    message = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "dense-fallback[reshape]" in message
    assert "not_implemented" in message


def test_sparse_flattening_reshape_logs_unrepresentable(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    seed = make_laplacian_input(
        jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
        sparse_axis=0,
    )

    forward_laplacian(lambda value: jnp.reshape(value, (6,)))(seed)

    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    message = record.getMessage()
    assert record.levelno == logging.DEBUG
    assert "dense-fallback[reshape]" in message
    assert "unrepresentable" in message


def test_mismatched_local1_maximum_logs_not_implemented(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    lhs, rhs = mismatched_local1_pair()

    forward_laplacian(jnp.maximum)(lhs, rhs)

    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    message = record.getMessage()
    assert record.levelno == logging.WARNING
    assert "dense-fallback[max]" in message
    assert "not_implemented" in message


@pytest.mark.parametrize(
    ("fn", "seed_factory", "indices", "expected_shape"),
    (
        pytest.param(
            operator.itemgetter(
                (
                    jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
                    jnp.zeros((2, 2), dtype=jnp.int32),
                )
            ),
            lambda: make_laplacian_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                sparse_axis=0,
            ),
            None,
            (9, 2, 2),
            id="checkerboard_owner",
        ),
        pytest.param(
            lambda value: jax.lax.gather(
                value,
                jnp.array([[0], [1]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, 2),
                    collapsed_slice_dims=(),
                    start_index_map=(0,),
                ),
                slice_sizes=(2, value.shape[1]),
            ),
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            None,
            (12, 2, 2, 3),
            id="moving_owner_window",
        ),
        pytest.param(
            operator.itemgetter(
                (
                    jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
                    jnp.array([[0, 0], [1, 1]], dtype=jnp.int32),
                )
            ),
            lambda: LapTuple(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                Local2Jacobian(
                    blocks=jnp.ones((2, 1, 3, 3), dtype=jnp.float32),
                    owners=OwnerRoles(
                        OwnerRole(0, np.arange(3, dtype=np.int32)),
                        OwnerRole(1, np.arange(3, dtype=np.int32)),
                    ),
                    input_shape=(3, 1),
                    input_owner_axis=0,
                ),
                jnp.zeros((3, 3), dtype=jnp.float32),
            ),
            None,
            (3, 2, 2),
            id="one_local2_role",
        ),
        pytest.param(
            lambda value, index: value[index],
            lambda: make_laplacian_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                sparse_axis=0,
            ),
            jnp.array([2, 0], dtype=jnp.int32),
            (9, 2, 3),
            id="owner_selecting",
        ),
        pytest.param(
            lambda value, start_indices: jax.lax.gather(
                value,
                start_indices[:, None],
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(1,),
                    start_index_map=(1,),
                ),
                slice_sizes=(0, 1),
            ),
            lambda: forward_laplacian(operator.itemgetter(slice(0)))(
                make_laplacian_input(
                    jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2),
                    sparse_axis=0,
                )
            ),
            jnp.array([0], dtype=jnp.int32),
            (6, 1, 0),
            id="empty_owner_role",
        ),
    ),
)
def test_unrepresentable_gather_falls_back(
    caplog,
    fn,
    seed_factory,
    indices,
    expected_shape,
):
    seed = seed_factory()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    transformed = forward_laplacian(fn)
    if indices is None:
        out = transformed(seed)
    else:
        out = jax.jit(lambda index: transformed(seed, index))(indices)

    assert isinstance(out.jacobian, jnp.ndarray)
    assert out.dense_jacobian.shape == expected_shape
    [record] = [
        record
        for record in caplog.records
        if "dense-fallback[gather]" in record.getMessage()
    ]
    assert record.levelno == logging.DEBUG
    assert "unrepresentable" in record.getMessage()
