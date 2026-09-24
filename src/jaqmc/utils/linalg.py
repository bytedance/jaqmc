# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""JAX linear-algebra helpers."""

import numpy as np
from jax import numpy as jnp

NDArray = jnp.ndarray | np.ndarray


def _slogdet_block(matrix: NDArray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``slogdet(matrix)``, treating an empty square matrix as identity.

    Raises:
        ValueError: If ``matrix`` is not square.
    """
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"Expected a square matrix. Got shape {matrix.shape}.")
    if matrix.shape[-1] == 0:
        leading = matrix.shape[:-2]
        return (
            jnp.ones(leading, dtype=matrix.dtype),
            jnp.zeros(leading, dtype=matrix.real.dtype),
        )
    return jnp.linalg.slogdet(matrix)


def slogdet_blocks(*matrices: NDArray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the signed log determinant of a block-diagonal matrix.

    ``matrices`` are the diagonal blocks. Empty square blocks contribute the
    identity determinant, avoiding ``jnp.linalg.slogdet`` on empty matrices.

    Args:
        *matrices: Square matrices with optional leading batch dimensions.

    Returns:
        ``(sign, log_abs_det)`` for the block-diagonal matrix. Leading batch
        dimensions follow JAX broadcasting rules.

    Raises:
        ValueError: If no blocks are supplied or a block is not square.
    """
    if not matrices:
        raise ValueError("Expected at least one matrix block.")

    sign, log_abs_det = _slogdet_block(matrices[0])
    for matrix in matrices[1:]:
        block_sign, block_log_abs_det = _slogdet_block(matrix)
        sign = sign * block_sign
        log_abs_det = log_abs_det + block_log_abs_det
    return sign, log_abs_det
