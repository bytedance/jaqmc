# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Tracked-input constructors for Forward Laplacian."""

from typing import Any, overload

import jax
import jax.flatten_util as jfu
import jax.tree_util as jtu
import numpy as np
from jax import numpy as jnp

from .sparse import Local1Jacobian, OwnerRole, OwnerRoles, canonical_axis
from .types import LapTuple


def make_local1_seed_jacobian(
    x: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,
    sparse_axis: int = 0,
) -> Local1Jacobian:
    """Build the initial ``Local1Jacobian`` for owner-local tracked inputs.

    The owner axis stays explicit in :class:`OwnerRole`, while all remaining input
    axes are flattened into the per-owner coordinate basis.

    Returns:
        A ``Local1Jacobian`` aligned with the input ownership map.
    """
    sparse_axis = canonical_axis(sparse_axis, x.ndim)
    if weights is None:
        weights = jnp.ones_like(x)
    weights = jnp.broadcast_to(weights, x.shape)
    input_shape = tuple(int(size) for size in x.shape)
    input_coord_shape = input_shape[:sparse_axis] + input_shape[sparse_axis + 1 :]
    input_coord_dim = int(np.prod(input_coord_shape, dtype=int))
    coord_indices = jnp.arange(input_coord_dim, dtype=jnp.int32).reshape(
        input_coord_shape
    )
    coord_indices = jnp.expand_dims(coord_indices, axis=sparse_axis)
    coord_indices = jnp.broadcast_to(coord_indices, x.shape)
    blocks = jnp.moveaxis(
        weights[..., None]
        * jax.nn.one_hot(
            coord_indices,
            input_coord_dim,
            dtype=weights.dtype,
        ),
        -1,
        0,
    )[None, ...]
    return Local1Jacobian(
        blocks=blocks,
        owners=OwnerRoles(
            OwnerRole(
                sparse_axis,
                np.arange(x.shape[sparse_axis], dtype=np.int32),
            )
        ),
        input_shape=input_shape,
        input_owner_axis=sparse_axis,
    )


@overload
def make_laplacian_input(
    x: Any, *, weights: Any = None, sparse_axis: None = None
) -> Any: ...
@overload
def make_laplacian_input(
    x: jnp.ndarray, *, weights: jnp.ndarray | None = None, sparse_axis: int
) -> LapTuple[Local1Jacobian]: ...
def make_laplacian_input(
    x: Any, *, weights: Any = None, sparse_axis: int | None = None
) -> Any | LapTuple[Local1Jacobian]:
    """Construct a tracked input ``LapTuple`` for Forward Laplacian.

    Args:
        x: Input array or dense pytree to seed for tracking.
        weights: Optional diagonal Jacobian weights broadcastable to ``x``.
            Dense mode accepts the same pytree structure as ``x``.
        sparse_axis: When ``None``, seed a dense identity Jacobian across the
            full flattened ``x`` pytree. When set, ``x`` must be a single array
            and the result is a sparse ``Local1`` Jacobian whose owner axis is
            ``sparse_axis`` and whose remaining axes are flattened into the
            tracked coordinate basis.

    Returns:
        A dense tree of ``LapTuple`` leaves or a sparse ``LapTuple`` suitable for
        :func:`jaqmc.laplacian.forward_laplacian`.

    Raises:
        TypeError: If sparse seeding is requested for a non-array input.
    """
    if sparse_axis is None:
        flat_x, unravel = jfu.ravel_pytree(x)
        identity = jnp.diag(jnp.ones_like(flat_x))
        jacobians = jax.vmap(unravel)(identity)
        laplacians = jtu.tree_map(jnp.zeros_like, x)
        seeds = jtu.tree_map(LapTuple, x, jacobians, laplacians)
        if weights is None:
            return seeds
        return jtu.tree_map(
            lambda seed, weight: LapTuple(
                seed.x,
                seed.jacobian * jnp.broadcast_to(weight, jnp.shape(seed.x)),
                seed.laplacian,
            ),
            seeds,
            weights,
            is_leaf=lambda value: isinstance(value, LapTuple),
        )
    if not isinstance(x, jnp.ndarray):
        raise TypeError("Sparse Laplacian input seeding requires a single array input.")
    return LapTuple(
        x,
        make_local1_seed_jacobian(x, weights=weights, sparse_axis=sparse_axis),
        jnp.zeros_like(x),
    )
