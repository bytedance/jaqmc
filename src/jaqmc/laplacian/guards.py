# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Type guards for Forward Laplacian container payloads."""

from typing import Any, TypeGuard

import jax.numpy as jnp

from .sparse import Local1Jacobian, Local2Jacobian, SparseJacobian
from .types import LapArgs, LapTuple


def is_sparse_jacobian(jacobian) -> TypeGuard[SparseJacobian]:
    """Return whether ``jacobian`` is a structured sparse Jacobian."""
    return isinstance(jacobian, Local1Jacobian | Local2Jacobian)


def is_laptuple(x: Any) -> TypeGuard[LapTuple[Any]]:
    """Return whether ``x`` is a ``LapTuple``."""
    return isinstance(x, LapTuple)


def is_dense_laptuple(x: Any) -> TypeGuard[LapTuple[jnp.ndarray]]:
    """Return whether ``x`` is a ``LapTuple`` carrying a dense Jacobian array."""
    return isinstance(x, LapTuple) and isinstance(x.jacobian, jnp.ndarray)


def dense_jacobian_needs_materialization(x: Any) -> bool:
    """Return whether a dense Jacobian must expand to ``(n, *x.shape)``."""
    return is_dense_laptuple(x) and x.jacobian.shape[1:] != x.x.shape


def is_local1_laptuple(x: Any) -> TypeGuard[LapTuple[Local1Jacobian]]:
    """Return whether ``x`` is a ``LapTuple`` carrying a ``Local1Jacobian``."""
    return isinstance(x, LapTuple) and isinstance(x.jacobian, Local1Jacobian)


def is_local2_laptuple(x: Any) -> TypeGuard[LapTuple[Local2Jacobian]]:
    """Return whether ``x`` is a ``LapTuple`` carrying a ``Local2Jacobian``."""
    return isinstance(x, LapTuple) and isinstance(x.jacobian, Local2Jacobian)


def is_sparse_laptuple(x: Any) -> TypeGuard[LapTuple[SparseJacobian]]:
    """Return whether ``x`` is a ``LapTuple`` carrying a sparse payload."""
    return isinstance(x, LapTuple) and is_sparse_jacobian(x.jacobian)


def lap_args_are_dense(x: LapArgs[Any]) -> TypeGuard[LapArgs[jnp.ndarray]]:
    """Return whether every tracked arg carries a dense Jacobian array."""
    return all(is_dense_laptuple(a) for a in x.arrays)
