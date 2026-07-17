# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for laplacian tests."""

from pathlib import Path

import jax
import pytest

from tests.jax_version import MIN_FORWARD_LAPLACIAN_JAX

LAPLACIAN_TEST_DIR = Path(__file__).resolve().parent
SKIP_FOR_OLDER_JAX = pytest.mark.skip(
    reason="tests/laplacian requires JAX >= 0.7.1",
)


def pytest_collection_modifyitems(items):
    if jax.__version_info__ >= MIN_FORWARD_LAPLACIAN_JAX:
        return

    for item in items:
        if item.path.resolve().is_relative_to(LAPLACIAN_TEST_DIR):
            item.add_marker(SKIP_FOR_OLDER_JAX)
