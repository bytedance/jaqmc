# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Structured sparse Jacobian storage types for Forward Laplacian.

Theory-to-storage overview
==========================

The dense Jacobian layout in this package has shape
``(prod(input_shape), *result_shape)``. The leading axis indexes tracked scalar
input coordinates, and each slice along that axis has the same shape as the
function output.

This file stores the same derivatives in a smaller layout when the Jacobian is
sparse in a very specific way: there is one chosen input axis, called
``input_owner_axis``, and each output element depends on coordinates from only
one or two entries of that axis.

Two common cases are:

- one-entry dependence: an output like ``f[..., i, ...]`` depends only on the
  input entry ``x[i]`` along ``input_owner_axis``;
- two-entry dependence: an output like ``f[..., i, ..., j, ...]`` depends only
  on the input entries ``x[i]`` and ``x[j]`` along ``input_owner_axis``.

``Local1Jacobian`` stores the one-entry case. ``Local2Jacobian`` stores the
two-entry case. If an operation would need an even richer dependency pattern,
the sparse representation is no longer exact and the code falls back to a dense
Jacobian.

Specifically, the sparse layout stores the following information:

- ``blocks[slot, :, *c]`` stores derivatives for output element ``(*c)``;
- ``owners[slot]`` says which row of ``input_owner_axis`` that slot refers to;
- axis ``1`` of ``blocks`` has length ``input_coord_dim``, which is the number
  of coordinates inside one row of ``input_owner_axis``.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import ClassVar, Self

import jax
import numpy as np
from jax import numpy as jnp

SPARSE_PREFIX_RANK = 2


def sparse_output_axes(blocks_ndim: int) -> tuple[int, ...]:
    """Return the block-axis indices that carry the public output shape."""
    return tuple(range(SPARSE_PREFIX_RANK, blocks_ndim))


@dataclass(frozen=True)
class OwnerRole:
    """Owner metadata for one sparse support slot.

    Attributes:
        axis: Identifies which output axis varies across the owner ids stored in
            ``values``. ``axis=None`` means the slot has a constant owner id, so
            every output element in that broadcasted region refers to the same
            original tracked owner id.
        values: Owner ids for the slot. Constant roles store exactly one owner
            id, while axis-varying roles store one owner id per entry of
            ``axis``.
    """

    axis: int | None
    values: np.ndarray

    def __post_init__(self):
        """Normalize constructor inputs to the canonical owner-role form.

        Raises:
            ValueError: If the role is not representable in the one-axis owner model.
        """
        values = np.array(self.values, dtype=np.int32, copy=True)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1:
            raise ValueError("OwnerRole values must be a 1D integer array.")
        axis = self.axis
        if axis is not None and axis < 0:
            raise ValueError("OwnerRole axis must be non-negative.")
        if values.size == 1:
            axis = None
        if axis is None and values.size > 1:
            raise ValueError("Constant OwnerRole must store exactly one owner id.")
        values.setflags(write=False)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "values", values)

    def reduce_output_axes(
        self, axes: tuple[int, ...], *, output_ndim: int
    ) -> Self | None:
        """Return owner role after summing over axes of the Jacobian output.

        Returns:
            Updated role, or ``None`` when the sparse model cannot represent the
            reduced result.
        """
        if self.axis is None:
            return self
        reduced_axes = canonical_axes(axes, output_ndim)
        if self.axis in reduced_axes:
            if self.values.size == 0 or np.all(self.values == self.values[:1]):
                return replace(self, axis=None, values=self.values[:1])
            return None
        return replace(
            self, axis=self.axis - sum(axis < self.axis for axis in reduced_axes)
        )

    def factorized_shape(self, output_ndim: int) -> tuple[int, ...]:
        """Return the broadcastable shape of this role over an output layout."""
        shape = [1] * output_ndim
        if self.axis is not None:
            shape[self.axis] = self.values.size
        return tuple(shape)

    def selector(self, input_n_particles: int, dtype, output_ndim: int) -> jnp.ndarray:
        """Return a broadcastable one-hot selector for this role."""
        return jnp.reshape(
            jax.nn.one_hot(
                jnp.asarray(self.values),
                input_n_particles,
                dtype=dtype,
            ),
            (*self.factorized_shape(output_ndim), input_n_particles),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OwnerRole):
            return NotImplemented
        return self.axis == other.axis and np.array_equal(self.values, other.values)


def canonical_axis(axis: int, ndim: int) -> int:
    """Normalize *axis* to ``[0, ndim)``.

    Returns:
        The non-negative axis index.

    Raises:
        ValueError: If *axis* does not identify one of the ``ndim`` dimensions.
    """
    normalized = axis if axis >= 0 else axis + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for ndim {ndim}.")
    return normalized


def canonical_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Normalize each axis index in *axes* to ``[0, ndim)``.

    Returns:
        A tuple of non-negative axis indices.
    """
    return tuple(canonical_axis(axis, ndim) for axis in axes)


def _broadcastable_to(
    actual_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    name: str,
):
    if len(actual_shape) != len(target_shape):
        raise ValueError(
            f"{name} rank mismatch: expected rank {len(target_shape)}, "
            f"got rank {len(actual_shape)}."
        )
    for axis, (actual, target) in enumerate(
        zip(actual_shape, target_shape, strict=True)
    ):
        if actual not in (1, target):
            raise ValueError(f"{name} axis {axis} must be 1 or {target}, got {actual}.")


@dataclass(frozen=True, init=False)
class OwnerRoles:
    """Ordered owner metadata, one :class:`OwnerRole` per sparse support slot."""

    roles: tuple[OwnerRole, ...]

    def __init__(self, *roles: OwnerRole):
        if not roles:
            raise ValueError("OwnerRoles requires at least one OwnerRole.")
        if any(not isinstance(role, OwnerRole) for role in roles):
            raise TypeError("OwnerRoles requires explicit OwnerRole inputs.")
        object.__setattr__(self, "roles", tuple(roles))

    def __iter__(self) -> Iterator[OwnerRole]:
        return iter(self.roles)

    def __len__(self) -> int:
        return len(self.roles)

    def __getitem__(self, index: int) -> OwnerRole:
        return self.roles[index]

    def factorized_shape(self, output_ndim: int) -> tuple[int, ...]:
        """Return the combined broadcastable owner shape for all slots."""
        shape = (1,) * output_ndim
        for role in self.roles:
            shape = np.broadcast_shapes(shape, role.factorized_shape(output_ndim))
        return shape

    def selectors(
        self, input_n_particles: int, dtype, output_ndim: int
    ) -> tuple[jnp.ndarray, ...]:
        """Return one one-hot selector per owner slot."""
        return tuple(
            role.selector(input_n_particles, dtype, output_ndim) for role in self.roles
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OwnerRoles):
            return NotImplemented
        if len(self.roles) != len(other.roles):
            return False
        return all(lhs == rhs for lhs, rhs in zip(self.roles, other.roles, strict=True))

    def map(self, fn: Callable[[OwnerRole], OwnerRole | None]) -> Self | None:
        """Apply ``fn`` to each slot, aborting when any slot becomes unsupported.

        Returns:
            Updated owner slots, or ``None`` when any transformed slot is unsupported.
        """
        mapped: list[OwnerRole] = []
        for role in self.roles:
            updated = fn(role)
            if updated is None:
                return None
            mapped.append(updated)
        return self.__class__(*mapped)

    def reduce_output_axes(
        self,
        axes: tuple[int, ...],
        *,
        output_shape: tuple[int, ...],
    ) -> Self | None:
        """Map each role through :meth:`OwnerRole.reduce_output_axes`.

        Returns:
            Updated owner roles, or ``None`` if any role becomes unsupported after
            the reduction.
        """
        ndim = len(output_shape)
        return self.map(lambda owner: owner.reduce_output_axes(axes, output_ndim=ndim))


@dataclass(frozen=True, init=False)
class _SparseJacobianBase:
    """Shared storage contract for structured sparse Jacobians.

    Attributes:
        blocks: Sparse block array shaped
            ``(support_slots, input_coord_dim, *output_shape)``. The leading two
            axes store the sparse support slots and coordinate derivatives for
            each output element. Trailing axes index output elements in the
            public output layout.
        owners: Owner metadata with one :class:`OwnerRole` per sparse support slot.
        input_shape: Original tracked input shape before flattening to the dense
            derivative basis.
        input_owner_axis: Axis of ``input_shape`` whose entries are tracked by the
            owner ids stored in ``owners``.
    """

    blocks: jnp.ndarray
    owners: OwnerRoles
    input_shape: tuple[int, ...]
    input_owner_axis: int

    _owner_role_count: ClassVar[int]

    def __init__(
        self,
        blocks: jnp.ndarray,
        owners: OwnerRoles,
        input_shape: tuple[int, ...],
        input_owner_axis: int,
    ):
        if not input_shape:
            raise ValueError(f"{type(self).__name__} requires a non-empty input shape.")
        if input_owner_axis < 0 or input_owner_axis >= len(input_shape):
            raise ValueError(
                f"{type(self).__name__} input_owner_axis must be in "
                f"[0, {len(input_shape)}), got {input_owner_axis}."
            )
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "input_owner_axis", input_owner_axis)

        if blocks.ndim < SPARSE_PREFIX_RANK:
            raise ValueError(
                f"{type(self).__name__} blocks must include the fixed "
                "(support, coord) prefix."
            )
        if blocks.shape[:2] != (self._owner_role_count, self.input_coord_dim):
            raise ValueError(
                f"{type(self).__name__} support/coord mismatch: expected "
                f"({self._owner_role_count}, {self.input_coord_dim}), got "
                f"{blocks.shape[:2]}"
            )
        object.__setattr__(self, "blocks", blocks)

        if not isinstance(owners, OwnerRoles):
            raise TypeError(f"{type(self).__name__} requires explicit OwnerRoles.")
        if len(owners) != self._owner_role_count:
            raise ValueError(
                f"{type(self).__name__} requires exactly {self._owner_role_count} "
                f"owner roles, got {len(owners)}."
            )
        all_values = np.concatenate([role.values for role in owners])
        if all_values.size > 0 and (
            all_values.min() < 0 or all_values.max() >= self.input_n_particles
        ):
            raise ValueError(
                f"{type(self).__name__} owners must be in [0, input_n_particles)."
            )
        _broadcastable_to(
            owners.factorized_shape(len(self.output_shape)),
            self.output_shape,
            name="OwnerRoles shape",
        )
        object.__setattr__(self, "owners", owners)

    @property
    def dtype(self):
        """Return the dtype of the stored sparse blocks."""
        return self.blocks.dtype

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the output shape shared by blocks and the primal value."""
        return self.blocks.shape[SPARSE_PREFIX_RANK:]

    @property
    def input_n_particles(self) -> int:
        """Return the size of the tracked owner axis."""
        return self.input_shape[self.input_owner_axis]

    @property
    def input_coord_shape(self) -> tuple[int, ...]:
        """Return the non-owner input axes flattened into ``input_coord_dim``."""
        return (
            self.input_shape[: self.input_owner_axis]
            + self.input_shape[self.input_owner_axis + 1 :]
        )

    @property
    def input_coord_dim(self) -> int:
        """Return the flattened size of the non-owner tracked coordinates."""
        return int(np.prod(self.input_coord_shape, dtype=int))

    def astype(self, dtype):
        """Return a copy with ``blocks`` cast to ``dtype``."""
        return self.with_blocks(self.blocks.astype(dtype))

    def with_blocks(
        self,
        blocks: jnp.ndarray,
        owners: OwnerRoles | None = None,
    ) -> Self:
        """Return a copy with updated sparse block data.

        The leading block axes are always the fixed ``(support, coord)`` prefix,
        and trailing axes define the public output shape. ``owners`` is optional
        because many updates preserve the same owner-slot layout.
        """
        return replace(
            self,
            blocks=blocks,
            owners=self.owners if owners is None else owners,
        )

    def _blocks_by_output(self) -> jnp.ndarray:
        """Return block payload with shape ``(*output_shape, input_coord_dim)``."""
        return jnp.moveaxis(self.blocks, 1, -1)

    def _owner_rows_to_dense_basis(self, owner_rows: jnp.ndarray) -> jnp.ndarray:
        """Convert owner-row contributions to the public dense Jacobian layout.

        ``owner_rows`` has shape ``(*output_shape, input_n_particles,
        input_coord_dim)``. It stores one accumulated derivative row for every
        owner along ``input_owner_axis``. Dense materialization unflattens
        ``input_coord_dim`` into the non-owner input axes, restores the original
        ``input_shape`` axis order, then flattens those input axes into the
        standard leading derivative basis.

        Returns:
            Dense Jacobian data with global derivative axis
            ``(n, *output_shape)``.
        """
        output_ndim = len(self.output_shape)
        owner_rows = owner_rows.reshape(
            *self.output_shape,
            self.input_n_particles,
            *self.input_coord_shape,
        )
        input_axis_order: list[int] = []
        next_coord_axis = output_ndim + 1
        for axis in range(len(self.input_shape)):
            if axis == self.input_owner_axis:
                input_axis_order.append(output_ndim)
                continue
            input_axis_order.append(next_coord_axis)
            next_coord_axis += 1
        dense_input_axes = (*input_axis_order, *range(output_ndim))
        owner_rows = jnp.transpose(owner_rows, dense_input_axes)
        return owner_rows.reshape(
            int(np.prod(self.input_shape, dtype=int)),
            *self.output_shape,
        )


@dataclass(frozen=True, init=False)
class Local1Jacobian(_SparseJacobianBase):
    """Sparse Jacobian for outputs that depend on one input owner.

    Use this class when each output element depends on exactly one index along
    ``input_owner_axis``.

    Storage layout:

    - ``blocks[0, :, ...]`` stores the derivatives for one output element.
    - ``owners[0]`` gives the index along ``input_owner_axis`` that those
      derivatives belong to.

    ``to_dense()`` converts this representation to the usual dense Jacobian
    shape ``(n, *output_shape)``.
    """

    _owner_role_count: ClassVar[int] = 1

    def to_dense(self) -> jnp.ndarray:
        """Expand to the dense ``(n, *output_shape)`` Jacobian contract.

        Returns:
            A dense Jacobian with the standard leading derivative basis.
        """
        (selector,) = self.owners.selectors(
            self.input_n_particles,
            self.blocks.dtype,
            len(self.output_shape),
        )
        blocks_by_output = self._blocks_by_output()
        owner_rows = selector[..., :, None] * blocks_by_output[..., None, :]
        return self._owner_rows_to_dense_basis(owner_rows)


@dataclass(frozen=True, init=False)
class Local2Jacobian(_SparseJacobianBase):
    """Sparse Jacobian for outputs that depend on two input owners.

    See ``Local1Jacobian`` for the basic storage layout. Use this class when
    each output element depends on two indices along ``input_owner_axis``
    instead of one, for example for pairwise outputs.
    """

    _owner_role_count: ClassVar[int] = 2

    def to_dense(self) -> jnp.ndarray:
        """Expand to the dense ``(n, *output_shape)`` Jacobian contract.

        Returns:
            A dense Jacobian with the standard leading derivative basis.
        """
        first_selector, second_selector = self.owners.selectors(
            self.input_n_particles,
            self.blocks.dtype,
            len(self.output_shape),
        )
        first_blocks = jnp.moveaxis(self.blocks[0], 0, -1)
        second_blocks = jnp.moveaxis(self.blocks[1], 0, -1)
        owner_rows = (
            first_selector[..., :, None] * first_blocks[..., None, :]
            + second_selector[..., :, None] * second_blocks[..., None, :]
        )
        return self._owner_rows_to_dense_basis(owner_rows)


type SparseJacobian = Local1Jacobian | Local2Jacobian
