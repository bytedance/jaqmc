# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared sparse operations used by primitive handlers."""

from collections.abc import Callable
from typing import assert_never, overload

import jax
from jax import numpy as jnp

from ..sparse import (
    SPARSE_PREFIX_RANK,
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    SparseJacobian,
)
from ..types import LapTuple


def sparse_factor_shape(out_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the broadcast shape for sparse block scale factors and masks."""
    return (1, 1, *out_shape)


def transform_sparse_jacobian(
    jacobian: SparseJacobian,
    *,
    operation: str,
    owner_transform: Callable[[OwnerRole], OwnerRole | None],
    block_transform: Callable[[jnp.ndarray], jnp.ndarray],
) -> SparseJacobian:
    # A sparse transform is valid only if the support blocks and every owner role
    # can be transformed coherently. If either piece stops being exact, callers
    # must abandon the sparse representation instead of approximating it.
    owners = jacobian.owners.map(owner_transform)
    if owners is None:
        raise ValueError(f"{operation} cannot preserve owner-local sparse structure")
    return jacobian.with_blocks(
        block_transform(jacobian.blocks),
        owners,
    )


@overload
def broadcast_sparse_jacobian(
    jacobian: Local1Jacobian, shape: tuple[int, ...]
) -> Local1Jacobian: ...


@overload
def broadcast_sparse_jacobian(
    jacobian: Local2Jacobian, shape: tuple[int, ...]
) -> Local2Jacobian: ...


def broadcast_sparse_jacobian(
    jacobian: SparseJacobian, shape: tuple[int, ...]
) -> SparseJacobian:
    if jacobian.output_shape == shape:
        return jacobian
    shift = len(shape) - len(jacobian.output_shape)
    broadcast_dimensions = tuple(range(shift, shift + len(jacobian.output_shape)))
    return transform_sparse_jacobian(
        jacobian,
        operation="broadcast",
        owner_transform=lambda owner: (
            owner if owner.axis is None else OwnerRole(owner.axis + shift, owner.values)
        ),
        block_transform=lambda blocks: jax.lax.broadcast_in_dim(
            blocks,
            (blocks.shape[0], blocks.shape[1], *shape),
            (0, 1, *(axis + 2 for axis in broadcast_dimensions)),
        ),
    )


def broadcast_sparse_laptuple(
    sparse: LapTuple,
    out_shape: tuple[int, ...],
) -> LapTuple:
    """Broadcast sparse value, Jacobian, and Laplacian to *out_shape*.

    Returns:
        A ``LapTuple`` with all three payloads broadcast to ``out_shape``.
    """
    jacobian = sparse.jacobian
    assert isinstance(jacobian, Local1Jacobian | Local2Jacobian)
    return LapTuple(
        jnp.broadcast_to(sparse.x, out_shape),
        broadcast_sparse_jacobian(jacobian, out_shape),
        jnp.broadcast_to(sparse.laplacian, out_shape),
    )


def broadcast_sparse_and_plain_to_common_shape(
    sparse: LapTuple[SparseJacobian],
    plain: jnp.ndarray,
) -> tuple[LapTuple[SparseJacobian], jnp.ndarray]:
    """Broadcast sparse and plain operands to their shared output shape.

    Returns:
        The sparse and plain operands broadcast to the same output shape.
    """
    out_shape = jnp.broadcast_shapes(sparse.x.shape, plain.shape)
    return broadcast_sparse_laptuple(sparse, out_shape), jnp.broadcast_to(
        plain, out_shape
    )


def compatible_sparse_metadata(lhs: SparseJacobian, rhs: SparseJacobian) -> bool:
    """Return whether two sparse payloads track the same input basis."""
    return lhs.input_shape == rhs.input_shape and (
        lhs.input_owner_axis == rhs.input_owner_axis
    )


def require_compatible_sparse_metadata(
    lhs: SparseJacobian,
    rhs: SparseJacobian,
    *,
    operation: str,
) -> None:
    """Raise when sparse Jacobian payloads disagree on tracked-input metadata.

    Raises:
        ValueError: If the tracked-input basis metadata differs.
    """
    if compatible_sparse_metadata(lhs, rhs):
        return
    raise ValueError(
        f"Incompatible {type(lhs).__name__} tracked-input metadata for {operation}: "
        f"lhs=(input_shape={lhs.input_shape}, "
        f"input_owner_axis={lhs.input_owner_axis}), "
        f"rhs=(input_shape={rhs.input_shape}, "
        f"input_owner_axis={rhs.input_owner_axis})."
    )


def zero_local_jacobian_like(
    template: SparseJacobian,
    shape: tuple[int, ...],
) -> SparseJacobian:
    """Return zeros with the same sparse metadata and payload family as ``template``."""
    zero = template.with_blocks(jnp.zeros_like(template.blocks))
    if zero.output_shape == shape:
        return zero
    return broadcast_sparse_jacobian(zero, shape)


def owner_ids_equal_mask(
    lhs: OwnerRole,
    rhs: OwnerRole,
    out_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Return where two sparse owner roles refer to the same input owner id."""
    lhs_owner_ids = jnp.broadcast_to(
        jnp.reshape(jnp.asarray(lhs.values), lhs.factorized_shape(len(out_shape))),
        out_shape,
    )
    rhs_owner_ids = jnp.broadcast_to(
        jnp.reshape(jnp.asarray(rhs.values), rhs.factorized_shape(len(out_shape))),
        out_shape,
    )
    return lhs_owner_ids == rhs_owner_ids


def lift_local1_binary_to_local2(
    lhs: Local1Jacobian,
    rhs: Local1Jacobian,
    *,
    rhs_sign: int = 1,
) -> Local2Jacobian:
    """Lift a binary Local1 combination into a two-particle local state.

    Returns:
        A ``Local2Jacobian`` carrying the two input-particle contributions.
    """
    require_compatible_sparse_metadata(lhs, rhs, operation="Local2 lift")
    return Local2Jacobian(
        blocks=jnp.stack(
            (lhs.blocks[0], rhs_sign * rhs.blocks[0]),
            axis=0,
        ),
        owners=OwnerRoles(lhs.owners[0], rhs.owners[0]),
        input_shape=lhs.input_shape,
        input_owner_axis=lhs.input_owner_axis,
    )


@overload
def scale_sparse_jacobian(
    jacobian: Local1Jacobian, factor: jnp.ndarray
) -> Local1Jacobian: ...


@overload
def scale_sparse_jacobian(
    jacobian: Local2Jacobian, factor: jnp.ndarray
) -> Local2Jacobian: ...


def scale_sparse_jacobian(
    jacobian: SparseJacobian, factor: jnp.ndarray
) -> SparseJacobian:
    """Scale any structured sparse Jacobian by an output-shaped factor.

    Returns:
        The scaled sparse Jacobian.
    """
    out_shape = jnp.broadcast_shapes(jacobian.output_shape, factor.shape)
    factor = jnp.broadcast_to(factor, out_shape).reshape(sparse_factor_shape(out_shape))
    if jacobian.output_shape == out_shape:
        blocks = jacobian.blocks
    else:
        blocks = jax.lax.broadcast_in_dim(
            jacobian.blocks,
            (jacobian.blocks.shape[0], jacobian.blocks.shape[1], *out_shape),
            (
                0,
                1,
                *(axis + 2 for axis in range(len(jacobian.output_shape))),
            ),
        )
    return jacobian.with_blocks(blocks * factor)


def _sparse_trace_jac_jac_T_or_H(
    jacobian: SparseJacobian, conjugate: bool = True
) -> jnp.ndarray:
    """Return ``tr(J J^T)`` or ``tr(J J^H)`` for a sparse Jacobian state.

    When ``conjugate`` is ``False`` this computes the algebraic contraction
    ``tr(J J^T)``, which may be complex for complex-valued Jacobians and must
    therefore preserve its imaginary part.

    When ``conjugate`` is ``True`` this computes the Hermitian contraction
    ``tr(J J^H)``. The matrix ``J J^H`` can have complex off-diagonal entries,
    but its trace is mathematically real:

    ``tr(J J^H) = sum_{i,k} |J_{ik}|^2``.
    """
    blocks = jacobian.blocks
    rhs_blocks = blocks.conj() if conjugate else blocks
    if isinstance(jacobian, Local1Jacobian):
        trace = jnp.sum(blocks * rhs_blocks, axis=(0, 1))
        return trace.real if conjugate else trace
    if isinstance(jacobian, Local2Jacobian):
        first_selector, second_selector = jacobian.owners.selectors(
            jacobian.input_n_particles,
            blocks.dtype,
            len(jacobian.output_shape),
        )
        block_norms = jnp.sum(blocks * rhs_blocks, axis=1)
        trace = jnp.sum(block_norms, axis=0)
        same_owner = jnp.sum(first_selector * second_selector, axis=-1) > 0
        cross = 2 * jnp.sum(blocks[0] * rhs_blocks[1], axis=0)
        total = trace + jnp.where(same_owner, cross, 0)
        return total.real if conjugate else total
    return assert_never(jacobian)


def sparse_trace_jac_jacT(jacobian: SparseJacobian) -> jnp.ndarray:
    """Return the algebraic trace ``tr(J J^T)`` for a sparse Jacobian.

    This is the contraction used in holomorphic chain rules, so the result may
    be complex and its imaginary part must be preserved.
    """
    return _sparse_trace_jac_jac_T_or_H(jacobian, conjugate=False)


def sparse_trace_jac_jacH(jacobian: SparseJacobian) -> jnp.ndarray:
    """Return the Hermitian trace ``tr(J J^H)`` for a sparse Jacobian.

    Although the full matrix ``J J^H`` may have complex off-diagonal entries,
    its trace is mathematically real and nonnegative:
    ``tr(J J^H) = sum_{i,k} |J_{ik}|^2``.
    """
    return _sparse_trace_jac_jac_T_or_H(jacobian, conjugate=True)


def sparse_output_block_axes(
    axes: tuple[int, ...],
    *,
    output_ndim: int,
) -> tuple[int, ...]:
    """Shift public output axes to block axes after the sparse prefix.

    Returns:
        Block-axis indices corresponding to the requested output axes.
    """
    canonical = tuple(axis if axis >= 0 else axis + output_ndim for axis in axes)
    return tuple(axis + SPARSE_PREFIX_RANK for axis in canonical)
