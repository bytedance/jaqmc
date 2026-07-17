# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Dense fallback kind and visibility policy for Forward Laplacian."""

import logging

import jax
import jax.numpy as jnp

from jaqmc.laplacian import (
    forward_laplacian,
    make_laplacian_input,
)
from jaqmc.laplacian.primitives.core import log_dense_fallback
from tests.laplacian.sparse.helpers import mismatched_local1_pair

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


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
