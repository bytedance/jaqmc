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


def _factorize_gathered_owner_labels(labels: np.ndarray) -> OwnerRole | None:
    """Factor gathered owner labels into JaQMC's one-axis owner model.

    ``-1`` labels identify invalid fill/drop windows.  Their derivative payload
    is zero, so they impose no owner constraint.  Each valid output position
    must agree with either one constant owner id or an owner id selected by one
    output axis.

    Returns:
        A factorized owner role, or ``None`` when more than one output axis is
        required to express the gathered labels.
    """
    if labels.ndim > 0 and any(size == 0 for size in labels.shape):
        return None

    valid = labels >= 0
    if not np.any(valid):
        return OwnerRole(None, np.array([0], dtype=np.int32))

    fallback_owner = int(labels[valid][0])
    if np.all(labels[valid] == fallback_owner):
        return OwnerRole(None, np.array([fallback_owner], dtype=np.int32))

    for axis, axis_size in enumerate(labels.shape):
        rows = np.moveaxis(labels, axis, 0).reshape(axis_size, -1)
        values = np.max(rows, axis=1, where=rows >= 0, initial=-1)
        values = np.where(values < 0, fallback_owner, values)
        if np.all((rows < 0) | (rows == values[:, None])):
            return OwnerRole(axis, values)
    return None


def _remap_gather_owner_role(
    owner: OwnerRole,
    *,
    operand_shape: tuple[int, ...],
    concrete_indices: np.ndarray | None,
    dimension_numbers,
    slice_sizes: tuple[int, ...],
    label_gather_kwargs,
) -> OwnerRole | None:
    """Transform one owner role through a gather.

    Returns:
        The remapped owner role, or ``None`` when the gather is not representable.
    """
    if concrete_indices is not None:
        if owner.axis is None:
            return owner
        labels = jnp.asarray(owner.values).reshape(
            owner.factorized_shape(len(operand_shape))
        )
        gathered = jax.lax.gather_p.bind(
            jnp.broadcast_to(labels, operand_shape),
            concrete_indices,
            **label_gather_kwargs,
        )
        return _factorize_gathered_owner_labels(np.asarray(gathered))

    if owner.values.size == 0:
        return None
    if owner.axis is None or np.all(owner.values == owner.values[0]):
        return OwnerRole(None, owner.values[:1])

    owner_axis = owner.axis
    if owner_axis in dimension_numbers.start_index_map:
        return None
    if owner_axis in dimension_numbers.operand_batching_dims:
        return None
    if owner_axis in dimension_numbers.collapsed_slice_dims:
        return OwnerRole(None, owner.values[:1])

    retained_dims = tuple(
        axis
        for axis in range(len(operand_shape))
        if axis not in dimension_numbers.collapsed_slice_dims
        and axis not in dimension_numbers.operand_batching_dims
    )
    output_axis = dimension_numbers.offset_dims[retained_dims.index(owner_axis)]
    return OwnerRole(output_axis, owner.values[: slice_sizes[owner_axis]])


def _gather_sparse_blocks(x: LapTuple, start_indices, kwargs) -> jnp.ndarray:
    slice_sizes = tuple(kwargs["slice_sizes"])
    dimension_numbers = kwargs["dimension_numbers"]
    support_shape = x.jacobian.blocks.shape[:2]
    block_dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(0, 1, *(axis + 2 for axis in dimension_numbers.offset_dims)),
        collapsed_slice_dims=tuple(
            axis + 2 for axis in dimension_numbers.collapsed_slice_dims
        ),
        start_index_map=tuple(axis + 2 for axis in dimension_numbers.start_index_map),
        operand_batching_dims=tuple(
            axis + 2 for axis in dimension_numbers.operand_batching_dims
        ),
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
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)

    slice_sizes = tuple(kwargs["slice_sizes"])
    dimension_numbers = kwargs["dimension_numbers"]
    if kwargs["mode"] == jax.lax.GatherScatterMode.ONE_HOT:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="not_implemented",
            reason="gather sparse ONE_HOT mode unsupported",
        )

    try:
        concrete_indices = jax.core.concrete_or_error(np.asarray, start_indices)
    except ConcretizationTypeError:
        concrete_indices = None
    label_gather_kwargs = {**kwargs, "fill_value": -1}
    owners = x.jacobian.owners.map(
        lambda owner: _remap_gather_owner_role(
            owner,
            operand_shape=x.x.shape,
            concrete_indices=concrete_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            label_gather_kwargs=label_gather_kwargs,
        )
    )
    if owners is None:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="gather sparse owner remap unsupported",
        )
    derivative_kwargs = {**kwargs, "fill_value": 0}
    y = jax.lax.gather_p.bind(x.x, start_indices, **kwargs)
    lapl = jax.lax.gather_p.bind(
        x.laplacian,
        start_indices,
        **derivative_kwargs,
    )
    blocks = _gather_sparse_blocks(x, start_indices, derivative_kwargs)
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
