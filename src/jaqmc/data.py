# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Self

import jax
import jax.core
from jax import numpy as jnp

from jaqmc.utils.jax_dataclass import JAXDataclassMeta
from jaqmc.utils.parallel_jax import BATCH_AXIS_NAME, all_gather


class Data(metaclass=JAXDataclassMeta):
    """Base container for structured wavefunction input data.

    :class:`Data` instances behave like lightweight, JAX-compatible dataclasses
    whose fields can be accessed both as attributes and as mapping keys. They
    are used to pass structured inputs (e.g. coordinates, atomic positions)
    between samplers, wavefunctions, and estimators.
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    @property
    def field_names(self) -> list[str]:
        """Return dataclass field names in declaration order."""
        return [field.name for field in dataclasses.fields(self)]

    def subset(self, fields: Sequence[str]) -> dict[str, Any]:
        """Return a dictionary containing only the selected fields.

        Args:
            fields: Sequence of field names to keep.

        Returns:
            A new ``dict`` mapping each requested field name to its value in
            this instance.

        Raises:
            KeyError: If any of the requested ``fields`` are not valid field
                names for this dataclass.
        """
        field_names = set(self.field_names)
        if missing := set(fields) - field_names:
            raise KeyError(
                "Requesting subset with fields "
                + ",".join(sorted(missing))
                + f", which are not present in {self.__class__.__name__} "
                f"({','.join(sorted(field_names))})"
            )
        return {k: getattr(self, k) for k in fields}

    def merge(self, values: Mapping[str, Any]) -> Self:
        """Return a new instance with ``values`` merged into this one.

        Args:
            values: Mapping from field names to replacement values.

        Returns:
            A new :class:`Data` (or subclass) instance where the provided
            ``values`` override the corresponding fields of this instance.

        Raises:
            KeyError: If any of the keys in ``values`` are not valid field
                names for this dataclass.
        """
        if extra := set(values) - set(self.field_names):
            raise KeyError(
                "Trying to merge values for unknown fields "
                + ",".join(sorted(extra))
                + f" into {self.__class__.__name__} "
                f"({','.join(sorted(self.field_names))})"
            )
        return dataclasses.replace(self, **values)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["data"],
    meta_fields=["fields_with_batch"],
)
@dataclasses.dataclass
class BatchedData[DataT: Data]:
    """Container pairing one data instance with batched-field metadata.

    Attributes:
        data: Structured runtime data. Batched fields keep the same dataclass
            structure as one-walker :class:`Data`, but carry an extra leading
            walker axis.
        fields_with_batch: Field names in ``data`` whose leaves carry the
            leading walker axis. Fields not listed here are shared across
            walkers.

    The type variable ``DataT`` is the concrete ``Data`` subtype stored in
    ``data``.
    """

    data: DataT
    fields_with_batch: Sequence[str]

    def check(self):
        """Validate the batched-field metadata against the wrapped data.

        The check verifies that every name in ``fields_with_batch`` exists on
        ``data`` and, for concrete JAX arrays, that every batched leaf shares
        the same leading batch size. Shape validation is skipped during JAX
        tracing and for non-array leaves.

        Raises:
            KeyError: If ``fields_with_batch`` names fields that do not exist
                on ``data``.
            ValueError: If a batched array leaf does not use the common
                leading batch size.
        """
        if not set(self.fields_with_batch) <= set(self.data.field_names):
            raise KeyError(
                "Requesting to batch over fields "
                + ",".join(set(self.fields_with_batch) - self.data.field_names)
                + f", which are not present in {self.data.__class__.__name__} "
                f"({','.join(self.data.field_names)})"
            )
        for leaf in jax.tree.leaves(self.data):
            if not isinstance(leaf, jax.Array) or isinstance(leaf, jax.core.Tracer):
                return  # JAX tracing
        batch_size = self.batch_size
        for key in self.fields_with_batch:
            for leaf in jax.tree.leaves(self.data[key]):
                if leaf.shape[0] != batch_size:
                    raise ValueError(
                        f"Data {key} {self.data[key].shape} is not properly "
                        f"batched with batch size {batch_size}."
                    )

    @property
    def batch_size(self) -> int:
        """Return the leading dimension shared by batched fields.

        The size is read from the first leaf of the first field listed in
        ``fields_with_batch``. Call :meth:`check` when you need to verify that
        all batched fields use the same leading size.

        Returns:
            The detected batch size, or ``0`` when no fields are marked as
            batched.
        """
        if not self.fields_with_batch:
            return 0
        first_batched_data = jax.tree.leaves(self.data[self.fields_with_batch[0]])
        return first_batched_data[0].shape[0]

    @property
    def vmap_axis(self):
        """Describe batch axes for use with :func:`jax.vmap`.

        Each dataclass field is mapped to an axis specification: ``0`` for
        fields that are batched along the leading dimension, and ``None`` for
        fields that are treated as broadcasted or scalar. The returned object
        is another instance of this dataclass, intended to be used as the
        ``in_axes`` argument to ``jax.vmap``.

        Returns:
            A new instance of the same dataclass with integer/``None`` values
            describing the batch axes per field.
        """
        return dataclasses.replace(
            self.data,
            **{
                field.name: 0 if field.name in self.fields_with_batch else None
                for field in dataclasses.fields(self.data)
            },
        )

    @property
    def partition_spec(self):
        """Describe how this batched data should be sharded.

        Fields listed in ``fields_with_batch`` are assigned a
        :class:`jax.sharding.PartitionSpec` over ``BATCH_AXIS_NAME``. Unbatched
        fields receive an empty partition spec, meaning they are shared rather
        than sharded over walkers.

        Returns:
            A new :class:`BatchedData` whose ``data`` fields contain partition
            specs matching the wrapped data structure.
        """
        return dataclasses.replace(
            self,
            data=dataclasses.replace(
                self.data,
                **{
                    field.name: jax.sharding.PartitionSpec(BATCH_AXIS_NAME)
                    if field.name in self.fields_with_batch
                    else jax.sharding.PartitionSpec()
                    for field in dataclasses.fields(self.data)
                },
            ),
        )

    def unbatched_example(self) -> DataT:
        """Return a one-walker-shaped example matching this data structure.

        Batched fields are replaced with arrays of ones after dropping their
        leading batch axis. Unbatched fields are replaced with ``ones_like``
        arrays of the same shape. The result is useful for initialization code
        that needs representative single-walker input shapes, not actual
        sampled values.

        Returns:
            A new ``DataT`` instance with the same fields as ``data`` and
            single-walker shapes for batched fields.
        """
        return dataclasses.replace(
            self.data,
            **{
                k: jax.tree.map(lambda x: jnp.ones(x.shape[1:]), self.data[k])
                for k in self.fields_with_batch
            },
            **{
                k: jax.tree.map(jnp.ones_like, self.data[k])
                for k in self.data.field_names
                if k not in self.fields_with_batch
            },
        )

    def fully_batched_data(self) -> DataT:
        """Return a new :class:`Data` with all fields batched.

        Fields that are already batched (listed in ``fields_with_batch``) are left
        unchanged; fields that are unbatched are duplicated along a new leading batch
        dimension so that their shapes become ``(batch_size, *orig_shape)``.

        Raises:
            ValueError: If the current ``batch_size`` is zero.
        """
        batch_size = self.batch_size
        if batch_size == 0:
            raise ValueError("Cannot fully batch data with batch size 0.")

        def _broadcast_leaf(x):
            return jnp.broadcast_to(x, (batch_size, *x.shape))

        return dataclasses.replace(
            self.data,
            **{
                name: jax.tree.map(_broadcast_leaf, self.data[name])
                for name in self.data.field_names
                if name not in self.fields_with_batch
            },
        )

    def all_gather(self) -> Self:
        """Gather distributed arrays from all devices to each local node.

        For fields that are batched (sharded along the batch axis), this
        collects all shards and materializes the complete array on each device.
        Unbatched fields are left unchanged.

        Returns:
            A new :class:`BatchedData` with all-gathered batched fields.
        """
        gathered = dataclasses.replace(
            self.data,
            **{
                name: jax.tree.map(all_gather, self.data[name])
                for name in self.data.field_names
                if name in self.fields_with_batch
            },
        )
        return dataclasses.replace(self, data=gathered)
