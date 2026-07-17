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

"""Forward Laplacian rules for shape-changing primitives."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive

from ..guards import (
    dense_jacobian_needs_materialization,
    is_dense_laptuple,
    is_local1_laptuple,
    is_local2_laptuple,
    is_sparse_laptuple,
)
from ..sparse import (
    OwnerRole,
    OwnerRoles,
    SparseJacobian,
    canonical_axes,
    canonical_axis,
)
from ..types import LaplacianHandler, LapTuple
from .core import (
    fallback_dense,
    wrap_linear,
)
from .sparse_ops import (
    require_compatible_sparse_metadata,
    sparse_output_block_axes,
    transform_sparse_jacobian,
)


def handle_transpose(args, kwargs):
    dense_handler = wrap_linear(
        lambda x, **params: jax.lax.transpose(x, tuple(params["permutation"]))
    )
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    permutation = tuple(kwargs["permutation"])
    return LapTuple(
        jax.lax.transpose(x.x, permutation),
        transform_sparse_jacobian(
            x.jacobian,
            operation="transpose",
            owner_transform=lambda owner: (
                owner
                if owner.axis is None
                else OwnerRole(permutation.index(owner.axis), owner.values)
            ),
            block_transform=lambda blocks: jnp.transpose(
                blocks,
                (0, 1, *(axis + 2 for axis in permutation)),
            ),
        ),
        jax.lax.transpose(x.laplacian, permutation),
    )


def _squeeze_owner_role(
    owner: OwnerRole,
    dimensions: tuple[int, ...],
) -> OwnerRole:
    if owner.axis is None:
        return owner
    if owner.axis in dimensions:
        return OwnerRole(None, owner.values[:1])
    return OwnerRole(
        owner.axis - sum(dim < owner.axis for dim in dimensions),
        owner.values,
    )


def _squeeze_sparse_jacobian(
    jacobian: SparseJacobian,
    dimensions: tuple[int, ...],
) -> SparseJacobian:
    dims = tuple(
        sorted(
            sparse_output_block_axes(
                dimensions,
                output_ndim=len(jacobian.output_shape),
            )
        )
    )
    output_dims = tuple(
        sorted(
            canonical_axes(dimensions, len(jacobian.output_shape)),
        )
    )
    return transform_sparse_jacobian(
        jacobian,
        operation="squeeze",
        owner_transform=lambda owner: _squeeze_owner_role(owner, output_dims),
        block_transform=lambda blocks: jnp.squeeze(blocks, axis=dims),
    )


def handle_squeeze(args, kwargs):
    dense_handler = wrap_linear(
        lambda x, **params: jax.lax.squeeze(x, tuple(params["dimensions"]))
    )
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    dimensions = tuple(kwargs["dimensions"])
    jacobian = _squeeze_sparse_jacobian(x.jacobian, dimensions)
    return LapTuple(
        jax.lax.squeeze(x.x, dimensions),
        jacobian,
        jax.lax.squeeze(x.laplacian, dimensions),
    )


def handle_rev(args, kwargs):
    dense_handler = wrap_linear(
        lambda x, **params: jax.lax.rev(x, tuple(params["dimensions"]))
    )
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    dimensions = tuple(kwargs["dimensions"])
    output_dims = tuple(
        axis if axis >= 0 else axis + len(x.jacobian.output_shape)
        for axis in dimensions
    )
    block_dims = sparse_output_block_axes(
        dimensions,
        output_ndim=len(x.jacobian.output_shape),
    )
    return LapTuple(
        jax.lax.rev(x.x, dimensions),
        transform_sparse_jacobian(
            x.jacobian,
            operation="rev",
            owner_transform=lambda owner: (
                owner
                if owner.axis is None or owner.axis not in output_dims
                else OwnerRole(owner.axis, owner.values[::-1])
            ),
            block_transform=lambda blocks: jax.lax.rev(blocks, block_dims),
        ),
        jax.lax.rev(x.laplacian, dimensions),
    )


def _slice_owner_role(
    owner: OwnerRole,
    start_indices: tuple[int, ...],
    limit_indices: tuple[int, ...],
    strides: tuple[int, ...],
) -> OwnerRole:
    if owner.axis is None:
        return owner
    axis = owner.axis
    return OwnerRole(
        axis,
        owner.values[start_indices[axis] : limit_indices[axis] : strides[axis]],
    )


def _slice_sparse_jacobian(
    jacobian: SparseJacobian,
    start_indices: tuple[int, ...],
    limit_indices: tuple[int, ...],
    strides: tuple[int, ...] | None,
) -> SparseJacobian:
    if strides is None:
        strides = (1,) * len(jacobian.output_shape)
    support_coord_shape = jacobian.blocks.shape[:2]
    # Slicing only restricts which output positions survive; it does not mix
    # support slots, so locality is preserved by slicing owners and blocks alike.
    return transform_sparse_jacobian(
        jacobian,
        operation="slice",
        owner_transform=lambda owner: _slice_owner_role(
            owner,
            start_indices,
            limit_indices,
            strides,
        ),
        block_transform=lambda blocks: jax.lax.slice(
            blocks,
            (0, 0, *start_indices),
            (*support_coord_shape, *limit_indices),
            (1, 1, *strides),
        ),
    )


def handle_slice(args, kwargs):
    dense_handler = wrap_linear(
        lambda x, **params: jax.lax.slice(
            x,
            tuple(params["start_indices"]),
            tuple(params["limit_indices"]),
            None if params.get("strides") is None else tuple(params["strides"]),
        )
    )
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    strides = kwargs.get("strides")
    start_indices = tuple(kwargs["start_indices"])
    limit_indices = tuple(kwargs["limit_indices"])
    strides = None if strides is None else tuple(strides)
    jacobian = _slice_sparse_jacobian(
        x.jacobian,
        start_indices,
        limit_indices,
        strides,
    )
    return LapTuple(
        jax.lax.slice(x.x, start_indices, limit_indices, strides),
        jacobian,
        jax.lax.slice(x.laplacian, start_indices, limit_indices, strides),
    )


def handle_split(args, kwargs) -> list[LapTuple]:
    dense_handler = wrap_linear(jax.lax.split_p.bind)
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)

    axis = int(kwargs["axis"])
    axis = axis if axis >= 0 else axis + x.x.ndim
    sizes = tuple(int(size) for size in kwargs["sizes"])
    ys = jax.lax.split_p.bind(x.x, sizes=sizes, axis=axis)
    lapls = jax.lax.split_p.bind(x.laplacian, sizes=sizes, axis=axis)
    jacobian_splits: list[SparseJacobian] = []
    start = 0
    output_shape = x.jacobian.output_shape
    rank = len(output_shape)
    for size in sizes:
        start_indices = [0] * rank
        limit_indices = list(output_shape)
        start_indices[axis] = start
        limit_indices[axis] = start + size
        start += size
        jacobian_splits.append(
            _slice_sparse_jacobian(
                x.jacobian,
                tuple(start_indices),
                tuple(limit_indices),
                None,
            )
        )
    return [
        LapTuple(y, jacobian, lapl)
        for y, jacobian, lapl in zip(ys, jacobian_splits, lapls, strict=True)
    ]


def _broadcast_in_dim_owner_role(
    owner: OwnerRole,
    broadcast_dimensions: tuple[int, ...],
) -> OwnerRole:
    if owner.axis is None:
        return owner
    return OwnerRole(broadcast_dimensions[owner.axis], owner.values)


def _broadcast_in_dim_sparse_jacobian(
    jacobian: SparseJacobian,
    shape: tuple[int, ...],
    broadcast_dimensions: tuple[int, ...],
) -> SparseJacobian:
    broadcast_dims_full = (0, 1, *(axis + 2 for axis in broadcast_dimensions))
    return transform_sparse_jacobian(
        jacobian,
        operation="broadcast_in_dim",
        owner_transform=lambda owner: _broadcast_in_dim_owner_role(
            owner, broadcast_dimensions
        ),
        block_transform=lambda blocks: jax.lax.broadcast_in_dim(
            blocks,
            (blocks.shape[0], blocks.shape[1], *shape),
            broadcast_dims_full,
        ),
    )


def _broadcast_in_dim_dense_jacobian(
    jacobian: jnp.ndarray,
    shape: tuple[int, ...],
    broadcast_dimensions: tuple[int, ...],
) -> jnp.ndarray:
    input_output_shape = jacobian.shape[1:]
    broadcasted_jacobian_shape = [1] * len(shape)
    for old_axis, new_axis in enumerate(broadcast_dimensions):
        broadcasted_jacobian_shape[new_axis] = input_output_shape[old_axis]
    return jax.lax.broadcast_in_dim(
        jacobian,
        (jacobian.shape[0], *broadcasted_jacobian_shape),
        (0, *(axis + 1 for axis in broadcast_dimensions)),
    )


def handle_broadcast_in_dim(args, kwargs):
    dense_handler = wrap_linear(
        lambda x, **params: jax.lax.broadcast_in_dim(
            x,
            tuple(params["shape"]),
            tuple(params["broadcast_dimensions"]),
        )
    )
    x = args[0]
    shape = tuple(kwargs["shape"])
    broadcast_dimensions = tuple(kwargs["broadcast_dimensions"])
    if is_dense_laptuple(x):
        return LapTuple(
            jax.lax.broadcast_in_dim(x.x, shape, broadcast_dimensions),
            _broadcast_in_dim_dense_jacobian(
                x.jacobian,
                shape,
                broadcast_dimensions,
            ),
            jax.lax.broadcast_in_dim(x.laplacian, shape, broadcast_dimensions),
        )
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    jacobian = _broadcast_in_dim_sparse_jacobian(
        x.jacobian,
        shape,
        broadcast_dimensions,
    )
    return LapTuple(
        jax.lax.broadcast_in_dim(x.x, shape, broadcast_dimensions),
        jacobian,
        jax.lax.broadcast_in_dim(x.laplacian, shape, broadcast_dimensions),
    )


def _reshape_owner_role(
    owner: OwnerRole,
    old_shape: tuple[int, ...],
    new_sizes: tuple[int, ...],
) -> OwnerRole | None:
    if owner.axis is None:
        return owner
    leading = int(np.prod(old_shape[: owner.axis], dtype=int))
    trailing = int(np.prod(old_shape[owner.axis + 1 :], dtype=int))
    prefix = 1
    preserved_axis: int | None = None
    # An owner role survives reshape only if its tracked axis becomes exactly one
    # new axis with the same cardinality and the same surrounding volume.
    for axis, size in enumerate(new_sizes):
        suffix = int(np.prod(new_sizes[axis + 1 :], dtype=int))
        if prefix == leading and size == owner.values.size and suffix == trailing:
            if preserved_axis is not None:
                return None
            preserved_axis = axis
        prefix *= size
    if preserved_axis is None:
        return None
    return OwnerRole(preserved_axis, owner.values)


def handle_reshape(args, kwargs):
    dense_handler = wrap_linear(
        lambda x, **params: jax.lax.reshape(
            x,
            tuple(params["new_sizes"]),
            dimensions=params.get("dimensions"),
        ),
        name="reshape",
    )
    x = args[0]
    if not is_sparse_laptuple(x):
        return dense_handler(args, kwargs)
    dimensions = kwargs.get("dimensions")
    if dimensions not in (None, tuple(range(x.x.ndim))):
        # This sparse rule only handles pure reshapes of the current output
        # order. Once reshape also permutes axes, the ownership question changes.
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="not_implemented",
            reason="reshape with non-identity dimensions",
        )
    new_sizes = tuple(kwargs["new_sizes"])
    owners = x.jacobian.owners.map(
        lambda owner: _reshape_owner_role(owner, x.jacobian.output_shape, new_sizes)
    )
    if owners is None:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="reshape cannot preserve sparse owner roles",
        )
    jacobian = x.jacobian.with_blocks(
        jnp.reshape(
            x.jacobian.blocks,
            (x.jacobian.blocks.shape[0], x.jacobian.blocks.shape[1], *new_sizes),
        ),
        owners=owners,
    )
    return LapTuple(
        jax.lax.reshape(x.x, new_sizes, dimensions=dimensions),
        jacobian,
        jax.lax.reshape(
            x.laplacian,
            new_sizes,
            dimensions=dimensions,
        ),
    )


def _concatenate_owner_role(
    owners: tuple[OwnerRole, ...],
    axis: int,
    segment_sizes: tuple[int, ...],
) -> OwnerRole | None:
    # Concatenation stays representable either when all inputs vary along the
    # concatenated axis, or when they are all constant/off-axis with identical
    # metadata that remains valid for the merged output.
    if all(owner.axis in (None, axis) for owner in owners):
        return OwnerRole(
            axis,
            np.concatenate(
                [
                    owner.values
                    if owner.axis == axis
                    else np.full(size, owner.values[0], dtype=np.int32)
                    for owner, size in zip(owners, segment_sizes, strict=True)
                ]
            ).astype(np.int32, copy=False),
        )
    if all(owner.axis != axis for owner in owners):
        result = owners[0]
        for owner in owners[1:]:
            if result != owner:
                return None
        return result
    return None


def _concatenate_owner_roles(
    owners: tuple[OwnerRoles, ...],
    axis: int,
    segment_sizes: tuple[int, ...],
) -> OwnerRoles | None:
    role_count = len(owners[0])
    if any(len(owner_set) != role_count for owner_set in owners[1:]):
        return None
    concatenated: list[OwnerRole] = []
    for role_index in range(role_count):
        role = _concatenate_owner_role(
            tuple(owner_set[role_index] for owner_set in owners),
            axis,
            segment_sizes,
        )
        if role is None:
            return None
        concatenated.append(role)
    return OwnerRoles(*concatenated)


def _concatenate_sparse_jacobians(
    jacobians: tuple[SparseJacobian | None, ...],
    shapes: tuple[tuple[int, ...], ...],
    axis: int,
) -> SparseJacobian | None:
    template = next(jac for jac in jacobians if jac is not None)
    assert template is not None
    axis = canonical_axis(axis, len(template.output_shape))
    blocks = []
    owners = []
    for jacobian, shape in zip(jacobians, shapes, strict=True):
        if jacobian is None:
            # Plain-array segments contribute zero derivatives, but can still
            # participate in a sparse concatenation if the owner metadata of the
            # sparse segments remains consistent after inserting zeros. This is
            # concatenate-specific shape construction, not broadcast: the plain
            # segment can be smaller or larger only along the concatenation axis.
            blocks.append(
                jnp.zeros_like(
                    template.blocks,
                    shape=(template.blocks.shape[0], template.blocks.shape[1], *shape),
                    dtype=template.blocks.dtype,
                )
            )
            owners.append(template.owners)
            continue
        if type(jacobian) is not type(template):
            raise ValueError("Incompatible sparse Jacobian families for concatenation.")
        require_compatible_sparse_metadata(jacobian, template, operation="concatenate")
        blocks.append(jacobian.blocks)
        owners.append(jacobian.owners)
    concatenated_blocks = jnp.concatenate(blocks, axis=axis + 2)
    concatenated_owners = _concatenate_owner_roles(
        tuple(owners),
        axis,
        tuple(shape[axis] for shape in shapes),
    )
    if concatenated_owners is None:
        return None
    return template.with_blocks(
        concatenated_blocks,
        concatenated_owners,
    )


def _concatenate_sparse_inputs(
    args,
    axis: int,
    *,
    family_name: str,
    dense_handler,
    kwargs,
):
    shapes = tuple(
        arg.x.shape if isinstance(arg, LapTuple) else arg.shape for arg in args
    )
    jacobians = tuple(
        arg.jacobian if isinstance(arg, LapTuple) else None for arg in args
    )
    jacobian = _concatenate_sparse_jacobians(jacobians, shapes, axis)
    if jacobian is None:
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason=f"concatenate {family_name} owner layout unsupported",
        )

    xs = tuple(arg.x if isinstance(arg, LapTuple) else arg for arg in args)
    lapls = tuple(
        arg.laplacian if isinstance(arg, LapTuple) else jnp.zeros_like(arg)
        for arg in args
    )
    return LapTuple(
        jax.lax.concatenate(xs, axis),
        jacobian,
        jax.lax.concatenate(lapls, axis),
    )


def handle_concatenate(args, kwargs):
    dense_handler = wrap_linear(
        lambda *xs, **params: jax.lax.concatenate(xs, params["dimension"]),
        name="concatenate",
    )
    axis = kwargs["dimension"]
    sparse_args = [is_sparse_laptuple(arg) for arg in args]
    if not any(sparse_args):
        if any(dense_jacobian_needs_materialization(arg) for arg in args):
            return fallback_dense(
                dense_handler,
                args,
                kwargs,
                kind="unrepresentable",
                reason="concatenate materializes compact dense Jacobians",
            )
        return dense_handler(args, kwargs)
    if any(is_dense_laptuple(arg) for arg in args):
        return fallback_dense(
            dense_handler,
            args,
            kwargs,
            kind="unrepresentable",
            reason="concatenate mixes sparse and dense LapTuple",
        )

    # Preserve one sparse family at a time. Mixing Local1 and Local2 inputs would
    # require an implicit coercion policy, so the handler rejects that case.
    if all(not isinstance(arg, LapTuple) or is_local2_laptuple(arg) for arg in args):
        return _concatenate_sparse_inputs(
            args,
            axis,
            family_name="Local2",
            dense_handler=dense_handler,
            kwargs=kwargs,
        )

    if all(not isinstance(arg, LapTuple) or is_local1_laptuple(arg) for arg in args):
        return _concatenate_sparse_inputs(
            args,
            axis,
            family_name="Local1",
            dense_handler=dense_handler,
            kwargs=kwargs,
        )

    return fallback_dense(
        dense_handler,
        args,
        kwargs,
        kind="not_implemented",
        reason="concatenate mixed sparse families",
    )


SHAPE_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.reshape_p: handle_reshape,
    jax.lax.transpose_p: handle_transpose,
    jax.lax.broadcast_in_dim_p: handle_broadcast_in_dim,
    jax.lax.slice_p: handle_slice,
    jax.lax.squeeze_p: handle_squeeze,
    jax.lax.concatenate_p: handle_concatenate,
    jax.lax.rev_p: handle_rev,
}
if hasattr(jax.lax, "split_p"):
    SHAPE_HANDLERS[jax.lax.split_p] = handle_split
