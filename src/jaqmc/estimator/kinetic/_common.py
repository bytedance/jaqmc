# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for kinetic energy estimators."""

from enum import StrEnum

from jax import numpy as jnp

from jaqmc.data import Data


class LaplacianMode(StrEnum):
    """Modes of calculating the diagonal Hessian for the Laplacian.

    Attributes:
        scan: Materializes all iterations via :py:func:`jax.lax.scan` —
            higher memory, faster compilation. Good default for small to
            medium systems.
        fori_loop: Runs one iteration at a time via
            :py:func:`jax.lax.fori_loop` — constant memory, slower
            compilation. Use when ``scan`` causes out-of-memory during
            compilation.
        forward_laplacian: Forward-mode Laplacian via
            `folx <https://github.com/microsoft/folx>`_. Can be fastest
            for large systems. Requires JAX >= 0.7.1.
    """

    scan = "scan"
    fori_loop = "fori_loop"
    forward_laplacian = "forward_laplacian"

    def __repr__(self) -> str:
        return str(self)


def _flatten_positions(
    data: Data, data_field: str
) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Validate and flatten position data for kinetic energy computation.

    Args:
        data: The data container.
        data_field: Name of the field containing positions.

    Returns:
        Tuple of (flattened_positions, original_shape).

    Raises:
        ValueError: If the data field is not a JAX array.
    """
    positions = data[data_field]
    if not isinstance(positions, jnp.ndarray):
        raise ValueError(
            f"Expected JAX Array for data field {data_field} for kinetic estimator."
            f" Got {type(data[data_field])}."
        )
    return positions.flatten(), positions.shape


def _apply_kinetic_formula(
    laplacian: jnp.ndarray, grad_squared: jnp.ndarray
) -> jnp.ndarray:
    """Apply kinetic energy formula: KE = -0.5 * (Laplacian + |grad|^2).

    Args:
        laplacian: The Laplacian of log(psi).
        grad_squared: The squared magnitude of gradient of log(psi).

    Returns:
        The kinetic energy.
    """
    return -0.5 * laplacian - 0.5 * grad_squared
