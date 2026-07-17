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

"""Core container and handler types for Forward Laplacian propagation."""

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp

from jaqmc.array_types import PyTree

from .sparse import SparseJacobian

type ExtraArgs = tuple[jnp.ndarray, ...]
type Arrays = tuple[jnp.ndarray, ...]


class LapTuple[JacobianT: jnp.ndarray | SparseJacobian](NamedTuple):
    """Triplet of (value, Jacobian, Laplacian) for Forward Laplacian propagation.

    The Laplacian has the same shape as ``x``. The Jacobian payload depends on
    where the ``LapTuple`` lives:

    - For dense paths, the Jacobian is a dense array with shape
      ``(n, *x_shape)``, where ``n`` is the total number of tracked scalar
      inputs. Axis ``0`` is the tracked-input / derivative basis.
    - For sparse-preserving paths, the same field may hold a structured payload
      from :mod:`jaqmc.laplacian.sparse`.

    Public callers should consume Jacobians through :attr:`dense_jacobian` when
    they need a uniform dense array. The ``jacobian`` field remains the storage
    slot used while the transform is propagating derivative state.

    The same structure also serves as the JAX pytree node for transform metadata
    such as ``vmap`` ``in_axes`` and ``out_axes``, or ``shard_map`` ``in_specs``
    and ``out_specs``. Build those trees with :meth:`pytree_spec` rather than
    passing derivative arrays to the constructor.
    """

    x: jnp.ndarray
    jacobian: JacobianT
    laplacian: jnp.ndarray

    @classmethod
    def pytree_spec(cls, x: Any, jacobian: Any, laplacian: Any) -> "LapTuple[Any]":
        """Return a ``LapTuple``-shaped JAX transform metadata tree.

        JAX requires ``in_axes``, ``out_axes``, and sharding spec trees to use
        the same pytree node type as the transformed value. For a function that
        returns a ``LapTuple``, each field needs its own axis or partition
        spec. Those values are typically integers, ``None``, or
        :class:`jax.sharding.PartitionSpec` objects—not derivative arrays. For
        batch axis ``a``, ``x`` and ``laplacian`` usually take ``a`` while
        ``jacobian`` takes ``a + 1`` because axis ``0`` is the derivative
        basis.

        Args:
            x: Axis or partition spec for the primal ``x`` field.
            jacobian: Axis or partition spec for the ``jacobian`` field.
            laplacian: Axis or partition spec for the ``laplacian`` field.

        Returns:
            A ``LapTuple`` whose fields hold the supplied metadata and can be
            passed to ``jax.vmap``, ``shard_map``, and similar transforms.
        """
        return cast(LapTuple[Any], cls(x, jacobian, laplacian))

    @property
    def shape(self):
        """Return the shape of the underlying array."""
        return self.x.shape

    @property
    def ndim(self):
        """Return the number of dimensions of the underlying array."""
        return self.x.ndim

    @property
    def size(self):
        """Return the total number of elements of the underlying array."""
        return self.x.size

    @property
    def dtype(self):
        """Return the dtype of the underlying array."""
        return self.x.dtype

    def astype(self, dtype):
        """Return a copy cast to the given dtype.

        Drops derivatives for non-float types.
        """
        if jax.dtypes.issubdtype(dtype, jnp.floating) or jax.dtypes.issubdtype(
            dtype, jnp.complexfloating
        ):
            return LapTuple(
                self.x.astype(dtype),
                self.jacobian.astype(dtype),
                self.laplacian.astype(dtype),
            )
        return self.x.astype(dtype)

    @property
    def dense_jacobian(self) -> jnp.ndarray:
        """Return the Jacobian materialized as a dense ``(n, *x.shape)`` array."""
        if not isinstance(self.jacobian, jnp.ndarray):
            return self.jacobian.to_dense()
        jacobian = cast(jnp.ndarray, self.jacobian)
        if jacobian.shape[1:] == self.x.shape:
            return jacobian
        if jacobian.ndim == 1:
            jacobian = jnp.reshape(
                jacobian,
                (jacobian.shape[0], *(1,) * self.x.ndim),
            )
        return jnp.broadcast_to(jacobian, (jacobian.shape[0], *self.x.shape))

    def to_dense(self) -> "LapTuple[jnp.ndarray]":
        return LapTuple(self.x, self.dense_jacobian, self.laplacian)


type ArrayOrLapTuple = jnp.ndarray | LapTuple[Any]
type LapTuples = tuple[LapTuple[Any], ...]
type DenseLapTuple = LapTuple[jnp.ndarray]


class LapArgs[JacobianT: jnp.ndarray | SparseJacobian](NamedTuple):
    """Utility wrapping a tuple of LapTuple values for convenient access."""

    arrays: tuple[LapTuple[JacobianT], ...]

    @property
    def x(self) -> Arrays:
        """Return the primal values from all contained LapTuples."""
        return tuple(a.x for a in self.arrays)

    @property
    def jacobian(self) -> tuple[JacobianT, ...]:
        """Return the Jacobian payloads from all contained LapTuples."""
        return tuple(a.jacobian for a in self.arrays)

    @property
    def laplacian(self) -> Arrays:
        """Return the Laplacians from all contained LapTuples."""
        return tuple(a.laplacian for a in self.arrays)

    def __len__(self) -> int:
        """Return the number of contained LapTuples."""
        return len(self.arrays)


Axes = Any
ForwardFn = Callable[..., Any]


class MergeFn(Protocol):
    """Protocol for functions that merge primitive args with extra args."""

    def __call__(self, args: Arrays, extra: ExtraArgs) -> Arrays:
        """Return merged arrays from tracked args and extra args."""
        ...


class LaplacianHandler(Protocol):
    """Protocol for Forward Laplacian handler functions."""

    def __call__(
        self,
        args: tuple[ArrayOrLapTuple, ...],
        kwargs: dict[str, Any],
    ) -> PyTree:
        """Return the Forward Laplacian result for the given args and kwargs."""
        ...
