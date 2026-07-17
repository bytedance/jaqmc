# Copyright 2023 Microsoft Corporation
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2026.
#
# Original file was released under MIT, with the full license text
# available at licenses/folx_MIT.txt
#
# This file is distributed under the Apache License 2.0,
# with portions originally licensed under the MIT License.

"""Forward Laplacian rules for indexing primitives."""

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.errors import ConcretizationTypeError
from jax.extend.core import Primitive

from ..guards import is_sparse_laptuple
from ..sparse import OwnerRole
from ..types import LaplacianHandler, LapTuple
from .core import fallback_dense, wrap_linear


def _supports_sparse_scalar_gather_layout(
    operand_ndim: int,
    start_indices,
    *,
    dimension_numbers,
    slice_sizes: tuple[int, ...],
) -> bool:
    if (
        dimension_numbers.offset_dims
        or dimension_numbers.operand_batching_dims
        or dimension_numbers.start_indices_batching_dims
    ):
        return False
    if len(slice_sizes) != operand_ndim or tuple(slice_sizes) != (1,) * operand_ndim:
        return False
    if tuple(dimension_numbers.collapsed_slice_dims) != tuple(range(operand_ndim)):
        return False
    start_index_map = tuple(dimension_numbers.start_index_map)
    if tuple(sorted(start_index_map)) != tuple(range(operand_ndim)):
        return False
    indices = np.asarray(start_indices)
    return indices.ndim == 2 and indices.shape[1] == len(start_index_map)


def _remap_gather_owner_role(
    owner: OwnerRole,
    *,
    output_axis: int,
    start_indices,
    start_index_map: tuple[int, ...],
) -> OwnerRole | None:
    if owner.axis is None:
        return owner
    try:
        gather_pos = start_index_map.index(owner.axis)
    except ValueError:
        return None
    indices = np.asarray(start_indices)[:, gather_pos]
    return OwnerRole(output_axis, owner.values[indices])


def _gather_sparse_blocks(x: LapTuple, start_indices, kwargs) -> jnp.ndarray:
    slice_sizes = tuple(kwargs["slice_sizes"])
    dimension_numbers = kwargs["dimension_numbers"]
    support_shape = x.jacobian.blocks.shape[:2]
    block_dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(0, 1),
        collapsed_slice_dims=tuple(
            axis + 2 for axis in dimension_numbers.collapsed_slice_dims
        ),
        start_index_map=tuple(axis + 2 for axis in dimension_numbers.start_index_map),
        operand_batching_dims=dimension_numbers.operand_batching_dims,
        start_indices_batching_dims=dimension_numbers.start_indices_batching_dims,
    )
    return jax.lax.gather_p.bind(
        x.jacobian.blocks,
        start_indices,
        dimension_numbers=block_dimension_numbers,
        slice_sizes=(*support_shape, *slice_sizes),
        unique_indices=kwargs["unique_indices"],
        indices_are_sorted=kwargs["indices_are_sorted"],
        mode=kwargs["mode"],
        fill_value=kwargs["fill_value"],
    )


def handle_gather(args, kwargs):
    dense_handler = wrap_linear(jax.lax.gather_p.bind)
    if len(args) != 2:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="not_implemented",
            reason="gather primitive signature unsupported",
        )
    x, start_indices = args
    if isinstance(start_indices, LapTuple):
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="gather tracked start_indices unsupported",
        )
    try:
        start_indices = np.asarray(
            jax.core.concrete_or_error(lambda value: value, start_indices)
        )
    except ConcretizationTypeError:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="gather sparse start_indices must be concrete",
        )
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)

    slice_sizes = tuple(kwargs["slice_sizes"])
    dimension_numbers = kwargs["dimension_numbers"]
    if not _supports_sparse_scalar_gather_layout(
        x.x.ndim,
        start_indices,
        dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes,
    ):
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="not_implemented",
            reason="gather sparse scalar layout unsupported",
        )
    owners = x.jacobian.owners.map(
        lambda owner: _remap_gather_owner_role(
            owner,
            output_axis=0,
            start_indices=start_indices,
            start_index_map=tuple(dimension_numbers.start_index_map),
        )
    )
    if owners is None:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="not_implemented",
            reason="gather sparse owner remap unsupported",
        )
    y = jax.lax.gather_p.bind(x.x, start_indices, **kwargs)
    lapl = jax.lax.gather_p.bind(x.laplacian, start_indices, **kwargs)
    blocks = _gather_sparse_blocks(x, start_indices, kwargs)
    return LapTuple(
        y,
        x.jacobian.with_blocks(
            blocks,
            owners=owners,
        ),
        lapl,
    )


INDEXING_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.gather_p: handle_gather,
    jax.lax.dynamic_slice_p: wrap_linear(jax.lax.dynamic_slice_p.bind),
    jax.lax.scatter_p: wrap_linear(jax.lax.scatter_p.bind),
    jax.lax.scatter_add_p: wrap_linear(jax.lax.scatter_add_p.bind),
    jax.lax.scatter_max_p: wrap_linear(jax.lax.scatter_max_p.bind),
    jax.lax.scatter_min_p: wrap_linear(jax.lax.scatter_min_p.bind),
}
