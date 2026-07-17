# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""JAX version predicates and pytest markers."""

import jax
import pytest

MIN_FORWARD_LAPLACIAN_JAX = (0, 7, 1)

requires_forward_laplacian = pytest.mark.skipif(
    jax.__version_info__ < MIN_FORWARD_LAPLACIAN_JAX,
    reason="forward_laplacian requires JAX >= 0.7.1",
)


def supports_forward_laplacian() -> bool:
    """Return whether this JAX version can run Forward Laplacian tests."""
    return jax.__version_info__ >= MIN_FORWARD_LAPLACIAN_JAX
