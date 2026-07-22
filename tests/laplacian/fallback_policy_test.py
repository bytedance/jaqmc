# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Dense fallback logging policy shared by Forward Laplacian handlers."""

import logging

import pytest

from jaqmc.laplacian.primitives.core import log_dense_fallback

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


@pytest.mark.parametrize(
    ("kind", "level"),
    (
        pytest.param("not_implemented", logging.WARNING, id="not_implemented"),
        pytest.param("unrepresentable", logging.DEBUG, id="unrepresentable"),
    ),
)
def test_dense_fallback_kind_controls_log_severity(kind, level, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    log_dense_fallback(
        site="policy-test",
        kind=kind,
        reason="free-form reason is not part of this contract",
    )

    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    assert record.levelno == level
    assert f"dense-fallback[policy-test] {kind}" in record.getMessage()
