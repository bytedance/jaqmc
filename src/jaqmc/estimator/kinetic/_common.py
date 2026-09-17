# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for kinetic energy estimators."""

from collections.abc import Callable
from enum import StrEnum

import jax
from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.utils import parallel_jax


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
            :mod:`jaqmc.laplacian`. Can be fastest for large systems.
            Requires JAX >= 0.7.1.
    """

    scan = "scan"
    fori_loop = "fori_loop"
    forward_laplacian = "forward_laplacian"

    def __repr__(self) -> str:
        return str(self)


def default_laplacian_mode() -> LaplacianMode:
    return (
        LaplacianMode.scan
        if jax.__version_info__ < (0, 7, 1)
        else LaplacianMode.forward_laplacian
    )


def require_forward_laplacian(mode: LaplacianMode) -> None:
    if mode == LaplacianMode.forward_laplacian and jax.__version_info__ < (0, 7, 1):
        raise RuntimeError(
            "JAX version too old to run jaqmc.laplacian. "
            "Please upgrade to JAX 0.7.1 or later."
        )


def hessian_diagonal_laplacian(
    jvp: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    mode: LaplacianMode,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sum Hessian diagonal entries against a coordinate basis.

    ``weights`` is ``None`` for the unweighted Euclidean Laplacian.

    Returns:
        The (optionally weighted) sum of Hessian diagonal entries.

    Raises:
        ValueError: If ``mode`` is not a Hessian-diagonal mode.
    """
    eye = parallel_jax.pvary(jnp.eye(n))
    if weights is None:
        weights = jnp.ones(n)
    if mode == LaplacianMode.scan:
        _, diagonal = jax.lax.scan(
            lambda i, _: (i + 1, jvp(eye[i])[i]), 0, None, length=n
        )
        return jnp.sum(weights * diagonal)
    if mode == LaplacianMode.fori_loop:
        return jax.lax.fori_loop(
            0, n, lambda i, val: val + weights[i] * jvp(eye[i])[i], 0.0
        )
    raise ValueError(f"Unsupported Hessian-diagonal Laplacian mode {mode}.")


def flatten_positions(
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


def apply_kinetic_formula(
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
