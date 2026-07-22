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

"""Forward Laplacian rules for linear algebra primitives."""

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from ..guards import is_sparse_laptuple
from ..sparse import (
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    SparseJacobian,
)
from ..types import ArrayOrLapTuple, LaplacianHandler, LapTuple
from .core import (
    fallback_dense,
    pack_laptuple,
)
from .sparse_ops import owner_ids_equal_mask, require_compatible_sparse_metadata

type Side = Literal["lhs", "rhs"]


@dataclass(frozen=True)
class DotGeneralAxes:
    """Describe one ``dot_general`` in a form the handler can query directly.

    JAX passes ``dimension_numbers`` as nested tuples. This helper turns that
    structure into named fields and small queries so the dense and sparse code
    can talk about contracted, batched, free, and output axes without
    re-decoding the raw tuple shape at every call site.
    """

    lhs_contract: tuple[int, ...]
    rhs_contract: tuple[int, ...]
    lhs_batch: tuple[int, ...]
    rhs_batch: tuple[int, ...]

    @classmethod
    def from_dimension_numbers(
        cls,
        dimension_numbers: jax.lax.DotDimensionNumbers,
    ) -> "DotGeneralAxes":
        """Build the normalized axis view from JAX's ``dimension_numbers``.

        Returns:
            A ``DotGeneralAxes`` instance with tuple-normalized axis metadata.
        """
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
        return cls(
            lhs_contract=tuple(lhs_contract),
            rhs_contract=tuple(rhs_contract),
            lhs_batch=tuple(lhs_batch),
            rhs_batch=tuple(rhs_batch),
        )

    @property
    def dimension_numbers(self):
        return (self.lhs_contract, self.rhs_contract), (self.lhs_batch, self.rhs_batch)

    def contract_axes(self, side: Side) -> tuple[int, ...]:
        """Return the operand axes that disappear into the contraction."""
        return self.lhs_contract if side == "lhs" else self.rhs_contract

    def batch_axes(self, side: Side) -> tuple[int, ...]:
        """Return the operand axes that survive as shared batch axes."""
        return self.lhs_batch if side == "lhs" else self.rhs_batch

    def free_axes(self, side: Side, ndim: int) -> tuple[int, ...]:
        """Return the operand axes that become side-specific output axes."""
        contract = self.contract_axes(side)
        batch = self.batch_axes(side)
        return tuple(i for i in range(ndim) if i not in batch + contract)

    def surviving_operand_axes(self, side: Side, ndim: int) -> tuple[int, ...]:
        """Return original operand axes after dropping only contracted axes.

        Sparse owner reduction removes contracted output axes first. This helper
        keeps the remaining axes in their original operand numbering so they can
        be mapped back into the final ``dot_general`` output layout.
        """
        contract = self.contract_axes(side)
        return tuple(i for i in range(ndim) if i not in contract)

    def output_axis(
        self,
        side: Side,
        operand_axis: int,
        *,
        lhs_ndim: int,
        rhs_ndim: int,
    ) -> int | None:
        """Map one operand axis to its position in the ``dot_general`` output.

        The output layout is always ``(*batch, *lhs_free, *rhs_free)``. If the
        requested axis is contracted away, the method returns ``None`` because
        that axis no longer exists in the result.

        Returns:
            The axis index in the ``dot_general`` output, or ``None`` when the
            operand axis is contracted away.
        """
        if operand_axis in self.contract_axes(side):
            return None
        batch_axes = self.batch_axes(side)
        if operand_axis in batch_axes:
            return batch_axes.index(operand_axis)
        lhs_free = self.free_axes("lhs", lhs_ndim)
        rhs_free = self.free_axes("rhs", rhs_ndim)
        if side == "lhs":
            return len(self.lhs_batch) + lhs_free.index(operand_axis)
        return len(self.lhs_batch) + len(lhs_free) + rhs_free.index(operand_axis)


@dataclass(frozen=True)
class OutputOwnerSlot:
    """Remember where one sparse support slot lands after remapping.

    ``input_slot`` points back to the slot in the incoming sparse Jacobian.
    ``owner`` is the same support role expressed in the output coordinate system
    of the current ``dot_general`` call.
    """

    input_slot: int
    owner: OwnerRole


@dataclass(frozen=True)
class SparseDotLayout:
    """Describe the sparse owner layout of a representable ``dot_general``.

    A sparse result is only exact when every incoming owner slot can be
    translated into output coordinates and the combined result still fits the
    Local1/Local2 model. This object stores the merged output owners together
    with the mapping from each operand's input slots to those output slots.
    """

    owners: OwnerRoles
    lhs_output_slots: tuple[int, ...]
    rhs_output_slots: tuple[int, ...]

    @classmethod
    def _map_owner_slots(
        cls,
        jacobian: SparseJacobian,
        axes: DotGeneralAxes,
        *,
        side: Side,
        lhs_ndim: int,
        rhs_ndim: int,
    ) -> tuple[OutputOwnerSlot, ...] | None:
        """Translate one operand's owner slots into ``dot_general`` output axes.

        The method first reduces owner metadata across contracted output axes and
        then remaps any surviving varying owner axis into its final output-axis
        position. It returns ``None`` when the sparse owner description stops
        being exact during either step.

        Returns:
            The remapped owner slots for this operand, or ``None`` when the
            operand cannot stay in the sparse owner-local model.
        """
        reduced = jacobian.owners.reduce_output_axes(
            axes.contract_axes(side),
            output_shape=jacobian.output_shape,
        )
        if reduced is None:
            return None
        operand_ndim = lhs_ndim if side == "lhs" else rhs_ndim
        surviving_axes = axes.surviving_operand_axes(side, operand_ndim)
        mapped: list[OutputOwnerSlot] = []
        for input_slot, owner in enumerate(reduced):
            if owner.axis is None:
                mapped.append(OutputOwnerSlot(input_slot=input_slot, owner=owner))
                continue
            # ``reduce_output_axes`` reports axes in the contracted-away layout;
            # dot_general output mapping still needs the original operand axis.
            if owner.axis >= len(surviving_axes):
                return None
            original_axis = surviving_axes[owner.axis]
            output_axis = axes.output_axis(
                side,
                original_axis,
                lhs_ndim=lhs_ndim,
                rhs_ndim=rhs_ndim,
            )
            if output_axis is None:
                return None
            mapped.append(
                OutputOwnerSlot(
                    input_slot=input_slot,
                    owner=OwnerRole(output_axis, owner.values),
                )
            )
        return tuple(mapped)

    @classmethod
    def _merge_owner_slots(
        cls,
        lhs_owner_slots: tuple[OutputOwnerSlot, ...],
        rhs_owner_slots: tuple[OutputOwnerSlot, ...],
    ) -> "SparseDotLayout | None":
        """Merge remapped owner roles into the result's support slots.

        Identical owner roles from the left and right operands share one output
        slot. If the merged result would need more than two distinct roles, the
        exact sparse representation would exceed Local2 and the method returns
        ``None``.

        Returns:
            The merged sparse layout, or ``None`` when the merged support would
            require more than two owner roles.
        """
        owners: list[OwnerRole] = []
        lhs_output_slots: list[int] = []
        rhs_output_slots: list[int] = []

        def output_slot(owner: OwnerRole) -> int:
            for index, existing in enumerate(owners):
                if existing == owner:
                    return index
            owners.append(owner)
            return len(owners) - 1

        for owner_slot in lhs_owner_slots:
            lhs_output_slots.append(output_slot(owner_slot.owner))
        for owner_slot in rhs_owner_slots:
            rhs_output_slots.append(output_slot(owner_slot.owner))

        if not 1 <= len(owners) <= 2:
            return None
        return cls(
            owners=OwnerRoles(*owners),
            lhs_output_slots=tuple(lhs_output_slots),
            rhs_output_slots=tuple(rhs_output_slots),
        )

    @classmethod
    def from_operands(
        cls,
        lhs: SparseJacobian,
        rhs: SparseJacobian,
        axes: DotGeneralAxes,
        *,
        lhs_ndim: int,
        rhs_ndim: int,
    ) -> "SparseDotLayout | None":
        """Build the sparse result layout when both operands are sparse.

        Returns:
            The merged sparse layout, or ``None`` when either operand cannot be
            remapped or the combined layout exceeds Local2.
        """
        lhs_slots = cls._map_owner_slots(
            lhs,
            axes,
            side="lhs",
            lhs_ndim=lhs_ndim,
            rhs_ndim=rhs_ndim,
        )
        if lhs_slots is None:
            return None
        rhs_slots = cls._map_owner_slots(
            rhs,
            axes,
            side="rhs",
            lhs_ndim=lhs_ndim,
            rhs_ndim=rhs_ndim,
        )
        if rhs_slots is None:
            return None
        return cls._merge_owner_slots(lhs_slots, rhs_slots)

    @classmethod
    def for_one_sparse_operand(
        cls,
        jacobian: SparseJacobian,
        axes: DotGeneralAxes,
        *,
        side: Side,
        lhs_ndim: int,
        rhs_ndim: int,
    ) -> "SparseDotLayout | None":
        """Build the sparse result layout when only one operand is sparse.

        Returns:
            The sparse layout for the output, or ``None`` when the sparse
            operand cannot be remapped exactly through the contraction.
        """
        owner_slots = cls._map_owner_slots(
            jacobian,
            axes,
            side=side,
            lhs_ndim=lhs_ndim,
            rhs_ndim=rhs_ndim,
        )
        if owner_slots is None:
            return None
        if side == "lhs":
            return cls._merge_owner_slots(owner_slots, ())
        return cls._merge_owner_slots((), owner_slots)


def _map_leading_axis_dot_general(
    payload: jnp.ndarray,
    primal: jnp.ndarray,
    *,
    payload_side: Side,
    dot_kwargs: dict[str, Any],
) -> jnp.ndarray:
    """Apply ``dot_general`` independently to each leading payload slice.

    Dense Jacobians use the leading axis for the global derivative basis;
    sparse blocks use it for local input coordinates. In both cases that axis
    survives in the result, so mapping the original primal operation expresses
    the same calculus and storage convention.

    Returns:
        One ``dot_general`` result per leading payload slice.
    """

    def apply_one_slice(payload_slice: jnp.ndarray) -> jnp.ndarray:
        lhs, rhs = (
            (payload_slice, primal)
            if payload_side == "lhs"
            else (primal, payload_slice)
        )
        return jax.lax.dot_general(lhs, rhs, **dot_kwargs)

    return jax.vmap(apply_one_slice)(payload)


def _contract_leading_axis_dot_general(
    lhs_payload: jnp.ndarray, rhs_payload: jnp.ndarray, *, dot_kwargs: dict[str, Any]
) -> jnp.ndarray:
    """Contract payload leading axes together with the primal dot axes.

    This computes a dense mixed derivative trace or a sparse block-pair trace
    directly, rather than materializing one result per leading payload slice
    and reducing it afterwards.

    Returns:
        The payload-leading-axis contraction in the primal output layout.
    """
    axes = DotGeneralAxes.from_dimension_numbers(dot_kwargs["dimension_numbers"])
    dot_kwargs = dot_kwargs | {
        "dimension_numbers": (
            (
                (*(axis + 1 for axis in axes.lhs_contract), 0),
                (*(axis + 1 for axis in axes.rhs_contract), 0),
            ),
            (
                tuple(axis + 1 for axis in axes.lhs_batch),
                tuple(axis + 1 for axis in axes.rhs_batch),
            ),
        ),
    }
    return jax.lax.dot_general(lhs_payload, rhs_payload, **dot_kwargs)


def _handle_dense_dot_general_with_one_tracked_operand(
    tracked: LapTuple,
    plain: jnp.ndarray,
    *,
    tracked_is_lhs: bool,
    dot_kwargs: dict[str, Any],
) -> LapTuple[jnp.ndarray]:
    """Apply ``dot_general`` when only one dense operand carries derivatives.

    The primitive is linear in either argument, so the value and Laplacian are
    ordinary ``dot_general`` calls and each dense Jacobian slice can be pushed
    through the same dot against the plain operand.

    Returns:
        A dense ``LapTuple`` for the ``dot_general`` result.
    """
    primal_lhs, primal_rhs = (
        (tracked.x, plain) if tracked_is_lhs else (plain, tracked.x)
    )
    lapl_lhs, lapl_rhs = (
        (tracked.laplacian, plain) if tracked_is_lhs else (plain, tracked.laplacian)
    )
    y = jax.lax.dot_general(primal_lhs, primal_rhs, **dot_kwargs)
    jacobian = _map_leading_axis_dot_general(
        tracked.dense_jacobian,
        plain,
        payload_side="lhs" if tracked_is_lhs else "rhs",
        dot_kwargs=dot_kwargs,
    )
    lapl = jax.lax.dot_general(lapl_lhs, lapl_rhs, **dot_kwargs)
    return LapTuple(y, jacobian, lapl)


def _handle_dense_dot_general_with_two_tracked_operands(
    lhs: LapTuple,
    rhs: LapTuple,
    *,
    dot_kwargs: dict[str, Any],
) -> ArrayOrLapTuple:
    """Handle the dense case where both operands contribute derivatives.

    ``dot_general`` is bilinear. Each linear Jacobian term maps the primal
    operation over its surviving derivative-basis axis, while the mixed trace
    contracts that axis directly with the primal contraction axes.

    Returns:
        The dense Forward-Laplacian result for the ``dot_general`` call.
    """
    y = jax.lax.dot_general(lhs.x, rhs.x, **dot_kwargs)
    lhs_jacobian = _map_leading_axis_dot_general(
        lhs.dense_jacobian,
        rhs.x,
        payload_side="lhs",
        dot_kwargs=dot_kwargs,
    )
    rhs_jacobian = _map_leading_axis_dot_general(
        rhs.dense_jacobian,
        lhs.x,
        payload_side="rhs",
        dot_kwargs=dot_kwargs,
    )
    grad_y = lhs_jacobian + rhs_jacobian
    mixed_trace = _contract_leading_axis_dot_general(
        lhs.dense_jacobian,
        rhs.dense_jacobian,
        dot_kwargs=dot_kwargs,
    )
    lapl_y = (
        jax.lax.dot_general(lhs.laplacian, rhs.x, **dot_kwargs)
        + jax.lax.dot_general(lhs.x, rhs.laplacian, **dot_kwargs)
        + 2 * mixed_trace
    )
    return pack_laptuple(y, grad_y, lapl_y)


def _handle_dense_dot_general(
    args: tuple[ArrayOrLapTuple, ...],
    *,
    dot_kwargs: dict[str, Any],
) -> ArrayOrLapTuple:
    """Dispatch the dense-only ``dot_general`` cases by tracked-operand count.

    Returns:
        Either the plain ``dot_general`` output or a dense ``LapTuple``,
        depending on which operands are tracked.
    """
    lhs, rhs = args
    if not isinstance(lhs, LapTuple) and not isinstance(rhs, LapTuple):
        return jax.lax.dot_general(lhs, rhs, **dot_kwargs)
    if isinstance(lhs, LapTuple) ^ isinstance(rhs, LapTuple):
        tracked, plain = (lhs, rhs) if isinstance(lhs, LapTuple) else (rhs, lhs)
        assert isinstance(tracked, LapTuple)
        assert not isinstance(plain, LapTuple)
        return _handle_dense_dot_general_with_one_tracked_operand(
            tracked,
            plain,
            tracked_is_lhs=isinstance(lhs, LapTuple),
            dot_kwargs=dot_kwargs,
        )
    assert isinstance(lhs, LapTuple)
    assert isinstance(rhs, LapTuple)
    return _handle_dense_dot_general_with_two_tracked_operands(
        lhs,
        rhs,
        dot_kwargs=dot_kwargs,
    )


def _build_sparse_dot_jacobian(
    *,
    blocks: jnp.ndarray,
    owners: OwnerRoles,
    template: SparseJacobian,
) -> SparseJacobian:
    """Build the sparse ``dot_general`` Jacobian from the merged owner layout.

    ``dot_general`` may change the number of distinct owner roles in the output.
    This helper chooses the matching sparse Jacobian family from that merged
    owner layout while copying the tracked-input basis from ``template``.

    Returns:
        A ``Local1Jacobian`` or ``Local2Jacobian`` matching ``owners``.

    Raises:
        RuntimeError: If ``owners`` has a role count outside the sparse
            Jacobian families supported here.
    """
    if len(owners) == 1:
        return Local1Jacobian(
            blocks=blocks,
            owners=owners,
            input_shape=template.input_shape,
            input_owner_axis=template.input_owner_axis,
        )
    if len(owners) == 2:
        return Local2Jacobian(
            blocks=blocks,
            owners=owners,
            input_shape=template.input_shape,
            input_owner_axis=template.input_owner_axis,
        )
    msg = f"internal error: expected 1 or 2 sparse owners, got {len(owners)}"
    raise RuntimeError(msg)


def _contract_block_slot_with_primal(
    blocks: jnp.ndarray,
    primal: jnp.ndarray,
    *,
    block_side: Side,
    dot_kwargs: dict[str, Any],
) -> jnp.ndarray:
    """Contract one sparse support slot against the other operand's primal data.

    Each sparse block stores one column per tracked input coordinate. The helper
    runs the same ``dot_general`` for every such column and keeps that input
    coordinate axis in the output block payload.

    Returns:
        The contracted sparse block payload for one output support slot.
    """
    return _map_leading_axis_dot_general(
        blocks,
        primal,
        payload_side=block_side,
        dot_kwargs=dot_kwargs,
    )


def _contract_block_slot_pair(
    lhs_blocks: jnp.ndarray,
    rhs_blocks: jnp.ndarray,
    *,
    dot_kwargs: dict[str, Any],
) -> jnp.ndarray:
    """Contract two sparse support slots and sum over tracked input coordinates.

    The result is the mixed trace term for one pair of owner slots in the
    sparse-sparse Laplacian formula.

    Returns:
        The mixed trace contribution for one left/right owner-slot pair.
    """
    return _contract_leading_axis_dot_general(
        lhs_blocks,
        rhs_blocks,
        dot_kwargs=dot_kwargs,
    )


def _handle_sparse_plain_dot(
    sparse: LapTuple[SparseJacobian],
    plain: jnp.ndarray,
    *,
    sparse_side: Side,
    axes: DotGeneralAxes,
    dot_kwargs: dict[str, Any],
) -> LapTuple[SparseJacobian] | None:
    """Propagate one sparse operand through ``dot_general`` against a plain array.

    The sparse path stays exact only when the operand's owner slots can be
    remapped into the output layout without leaving the Local1/Local2 model.
    When that succeeds, each sparse block slot is contracted against the plain
    operand and written into the merged output-owner layout.

    Returns:
        A sparse ``LapTuple`` when the owner-local structure survives exactly,
        or ``None`` when this handler cannot represent the result sparsely.
    """
    layout = SparseDotLayout.for_one_sparse_operand(
        sparse.jacobian,
        axes,
        side=sparse_side,
        lhs_ndim=sparse.x.ndim if sparse_side == "lhs" else plain.ndim,
        rhs_ndim=plain.ndim if sparse_side == "lhs" else sparse.x.ndim,
    )
    if layout is None:
        return None

    lhs, rhs = (sparse.x, plain) if sparse_side == "lhs" else (plain, sparse.x)
    lapl_lhs, lapl_rhs = (
        (sparse.laplacian, plain) if sparse_side == "lhs" else (plain, sparse.laplacian)
    )
    y = jax.lax.dot_general(lhs, rhs, **dot_kwargs)
    out_blocks = jnp.zeros_like(
        y,
        shape=(len(layout.owners), sparse.jacobian.input_coord_dim, *y.shape),
        dtype=jnp.result_type(sparse.jacobian.dtype, plain.dtype, y.dtype),
    )
    output_slots = (
        layout.lhs_output_slots if sparse_side == "lhs" else layout.rhs_output_slots
    )
    for input_slot, output_slot in enumerate(output_slots):
        contribution = _contract_block_slot_with_primal(
            sparse.jacobian.blocks[input_slot],
            plain,
            block_side=sparse_side,
            dot_kwargs=dot_kwargs,
        )
        out_blocks = out_blocks.at[output_slot, :, ...].add(contribution)
    jacobian = _build_sparse_dot_jacobian(
        blocks=out_blocks,
        owners=layout.owners,
        template=sparse.jacobian,
    )
    lapl = jax.lax.dot_general(lapl_lhs, lapl_rhs, **dot_kwargs)
    return LapTuple(y, jacobian, lapl)


def _handle_sparse_sparse_dot(
    lhs: LapTuple[SparseJacobian],
    rhs: LapTuple[SparseJacobian],
    *,
    axes: DotGeneralAxes,
    dot_kwargs: dict[str, Any],
) -> LapTuple[SparseJacobian] | None:
    """Propagate two sparse operands through ``dot_general`` when possible.

    Both sparse Jacobians must describe the same tracked-input basis, and their
    remapped owner slots must merge into at most two distinct output roles. The
    resulting Laplacian combines the usual linear terms with a mixed trace term
    that is added only where the left and right owner ids refer to the same
    tracked input owner.

    Returns:
        A sparse ``LapTuple`` when both operands stay representable in the
        owner-local model, or ``None`` when the sparse layout breaks down.
    """
    require_compatible_sparse_metadata(
        lhs.jacobian, rhs.jacobian, operation="dot_general"
    )
    layout = SparseDotLayout.from_operands(
        lhs.jacobian,
        rhs.jacobian,
        axes,
        lhs_ndim=lhs.x.ndim,
        rhs_ndim=rhs.x.ndim,
    )
    if layout is None:
        return None

    y = jax.lax.dot_general(lhs.x, rhs.x, **dot_kwargs)
    out_blocks = jnp.zeros_like(
        y,
        shape=(len(layout.owners), lhs.jacobian.input_coord_dim, *y.shape),
        dtype=jnp.result_type(lhs.jacobian.dtype, rhs.jacobian.dtype, y.dtype),
    )

    for input_slot, output_slot in enumerate(layout.lhs_output_slots):
        contribution = _contract_block_slot_with_primal(
            lhs.jacobian.blocks[input_slot],
            rhs.x,
            block_side="lhs",
            dot_kwargs=dot_kwargs,
        )
        out_blocks = out_blocks.at[output_slot, :, ...].add(contribution)

    for input_slot, output_slot in enumerate(layout.rhs_output_slots):
        contribution = _contract_block_slot_with_primal(
            rhs.jacobian.blocks[input_slot],
            lhs.x,
            block_side="rhs",
            dot_kwargs=dot_kwargs,
        )
        out_blocks = out_blocks.at[output_slot, :, ...].add(contribution)

    mixed_trace = jnp.zeros_like(y, dtype=jnp.result_type(lhs.x.dtype, rhs.x.dtype))
    for lhs_input_slot, lhs_output_slot in enumerate(layout.lhs_output_slots):
        lhs_owner = layout.owners[lhs_output_slot]
        lhs_blocks = lhs.jacobian.blocks[lhs_input_slot]
        for rhs_input_slot, rhs_output_slot in enumerate(layout.rhs_output_slots):
            rhs_owner = layout.owners[rhs_output_slot]
            pair_trace = _contract_block_slot_pair(
                lhs_blocks,
                rhs.jacobian.blocks[rhs_input_slot],
                dot_kwargs=dot_kwargs,
            )
            mixed_trace = mixed_trace + jnp.where(
                owner_ids_equal_mask(lhs_owner, rhs_owner, y.shape),
                pair_trace,
                0,
            )

    jacobian = _build_sparse_dot_jacobian(
        blocks=out_blocks,
        owners=layout.owners,
        template=lhs.jacobian,
    )
    lapl = (
        jax.lax.dot_general(lhs.laplacian, rhs.x, **dot_kwargs)
        + jax.lax.dot_general(lhs.x, rhs.laplacian, **dot_kwargs)
        + 2 * mixed_trace
    )
    return LapTuple(y, jacobian, lapl)


def handle_dot_general(args: tuple[ArrayOrLapTuple, ...], kwargs: dict[str, Any]):
    """Forward-Laplacian rule for ``jax.lax.dot_general``.

    The handler prefers exact sparse retention when the owner-local Jacobian
    structure survives the contraction. When it does not, it either falls back
    to the dense implementation or reports that the sparse transform is not
    representable, depending on why the sparse path failed.

    Returns:
        The plain array output or ``LapTuple`` result for this ``dot_general``
        invocation, using sparse retention when it remains exact.
    """
    lhs, rhs = args
    axes = DotGeneralAxes.from_dimension_numbers(kwargs["dimension_numbers"])
    dot_kwargs = dict(kwargs)
    lhs_sparse = is_sparse_laptuple(lhs)
    rhs_sparse = is_sparse_laptuple(rhs)

    if not lhs_sparse and not rhs_sparse:
        return _handle_dense_dot_general(args, dot_kwargs=dot_kwargs)

    if lhs_sparse and rhs_sparse:
        assert isinstance(lhs, LapTuple)
        assert isinstance(rhs, LapTuple)
        result = _handle_sparse_sparse_dot(
            lhs,
            rhs,
            axes=axes,
            dot_kwargs=dot_kwargs,
        )
        if result is not None:
            return result
        return fallback_dense(
            handle_dot_general,
            args,
            kwargs,
            kind="unrepresentable",
            reason="dot_general sparse-sparse transform unsupported",
        )

    sparse, plain = (lhs, rhs) if lhs_sparse else (rhs, lhs)
    sparse_side: Side = "lhs" if lhs_sparse else "rhs"
    if isinstance(plain, LapTuple):
        return fallback_dense(
            handle_dot_general,
            args,
            kwargs,
            kind="unrepresentable",
            reason="dot_general sparse with dense LapTuple",
        )
    assert isinstance(sparse, LapTuple)
    result = _handle_sparse_plain_dot(
        sparse,
        plain,
        sparse_side=sparse_side,
        axes=axes,
        dot_kwargs=dot_kwargs,
    )
    if result is not None:
        return result
    return fallback_dense(
        handle_dot_general,
        args,
        kwargs,
        kind="unrepresentable",
        reason="dot_general sparse transform unsupported",
    )


DOT_GENERAL_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.dot_general_p: handle_dot_general,
}
