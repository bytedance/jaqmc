# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward Laplacian transform API for JAX.

The interpreter may carry structured sparse Jacobian states (see
``jaqmc.laplacian.sparse``): ``Local1`` tracks outputs that still depend on a
single original input particle block, and ``Local2`` tracks pairwise particle
streams. Public ``forward_laplacian(...)`` calls may return those payloads when
the sparse path survives to the output. Callers that need a dense array should
read the Jacobian through ``LapTuple.dense_jacobian`` and can seed tracked
inputs with ``make_laplacian_input(...)``.
"""

from .custom_rules import custom_laplacian
from .guards import (
    is_dense_laptuple,
    is_laptuple,
    is_local1_laptuple,
    is_local2_laptuple,
    is_sparse_jacobian,
    is_sparse_laptuple,
)
from .interpreter import forward_laplacian
from .primitives import AutoLaplacianFallback
from .seed import make_laplacian_input
from .sparse import (
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    SparseJacobian,
)
from .types import ArrayOrLapTuple, LapTuple

__all__ = [
    "ArrayOrLapTuple",
    "AutoLaplacianFallback",
    "LapTuple",
    "Local1Jacobian",
    "Local2Jacobian",
    "OwnerRole",
    "OwnerRoles",
    "SparseJacobian",
    "custom_laplacian",
    "forward_laplacian",
    "is_dense_laptuple",
    "is_laptuple",
    "is_local1_laptuple",
    "is_local2_laptuple",
    "is_sparse_jacobian",
    "is_sparse_laptuple",
    "make_laplacian_input",
]
