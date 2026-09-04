# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Source-pool batching, persistence, and reduction helpers for LIT."""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from upath import UPath

from jaqmc.data import BatchedData
from jaqmc.utils import parallel_jax
from jaqmc_contrib_lit.common import _save_npz
from jaqmc_contrib_lit.response import (
    LITSourceSums,
    LITStats,
    merge_source_sums,
    merge_source_sums_across_devices,
    stats_from_source_sums,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


def _flatten_batched_tree(tree, batch_size: int) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        msg = "Cannot build SR score matrix from an empty parameter tree."
        raise ValueError(msg)
    return jnp.concatenate(
        [jnp.reshape(leaf, (batch_size, -1)) for leaf in leaves],
        axis=1,
    )


def _concat_batched_data(pool):
    if not pool:
        msg = "At least one sampled batch is required to build a source pool."
        raise ValueError(msg)
    first = pool[0]
    updates = {}
    for field_name in first.fields_with_batch:
        updates[field_name] = jnp.concatenate(
            [getattr(batch.data, field_name) for batch in pool],
            axis=0,
        )
    return first.__class__(
        data=first.data.merge(updates),
        fields_with_batch=first.fields_with_batch,
    )


def _replicate_across_local_devices(value):
    """Place a pytree with replicated sharding on the local device mesh.

    Returns:
        The same logical value with ``PartitionSpec()`` sharding.
    """
    return jax.device_put(
        value,
        parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION),
    )


def _shard_rng_across_local_devices(rng):
    """Derive and shard one independent PRNG key per local device.

    Returns:
        A flattened key array partitioned so each device receives one key.
    """
    per_device_rngs = jax.random.split(rng, jax.local_device_count()).flatten()
    return jax.device_put(
        per_device_rngs,
        parallel_jax.make_sharding(parallel_jax.DATA_PARTITION),
    )


def _shard_batched_data_across_local_devices(pool: BatchedData) -> BatchedData:
    """Place batch fields across the local mesh and replicate shared fields.

    Returns:
        The unchanged global batch with its declared partitioning applied.
    """
    return jax.device_put(
        pool,
        parallel_jax.make_sharding(pool.partition_spec),
    )


def _slice_batched_data(pool: BatchedData, start: int, size: int) -> BatchedData:
    if size < 1:
        msg = "BatchedData chunk size must be positive."
        raise ValueError(msg)
    if start < 0 or start + size > pool.batch_size:
        msg = (
            f"Invalid BatchedData slice start={start} size={size} "
            f"for batch_size={pool.batch_size}."
        )
        raise ValueError(msg)
    batch_slice = slice(start, start + size)

    def slice_leaf(value):
        if isinstance(value, jax.Array) and isinstance(
            value.sharding,
            jax.sharding.NamedSharding,
        ):
            # Explicitly preserve the input NamedSharding.  Plain indexing is
            # ambiguous for a proper subset of a batch-partitioned global
            # array and raises ShardingTypeError on more than one device.
            return value.at[batch_slice].get(out_sharding=value.sharding)
        return operator.itemgetter(batch_slice)(value)

    updates = {
        field_name: jax.tree.map(
            slice_leaf,
            getattr(pool.data, field_name),
        )
        for field_name in pool.fields_with_batch
    }
    return pool.__class__(
        data=pool.data.merge(updates),
        fields_with_batch=pool.fields_with_batch,
    )


def _cyclic_batched_data_chunk(
    pool: BatchedData,
    requested_size: int,
    batch_index: int,
) -> BatchedData:
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    if chunk_size >= pool.batch_size:
        return pool
    chunk_count = max(1, pool.batch_size // chunk_size)
    start = (int(batch_index) % chunk_count) * chunk_size
    return _slice_batched_data(pool, start, chunk_size)


def _indexed_batched_data_chunk(
    pool: BatchedData,
    requested_size: int,
    chunk_index: int,
) -> BatchedData:
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    if chunk_size >= pool.batch_size:
        return pool
    if pool.batch_size % chunk_size != 0:
        raise ValueError(
            f"Source pool size {pool.batch_size} must be divisible by training "
            f"chunk size {chunk_size}."
        )
    chunk_count = pool.batch_size // chunk_size
    if not 0 <= int(chunk_index) < chunk_count:
        raise ValueError(
            f"Training chunk index {chunk_index} is outside [0, {chunk_count})."
        )
    return _slice_batched_data(pool, int(chunk_index) * chunk_size, chunk_size)


def _shuffled_batched_data_chunk_index(
    pool_size: int,
    requested_size: int,
    iteration: int,
    *,
    seed: int,
) -> int:
    """Map an iteration to a deterministic, epoch-wise random chunk index.

    Every epoch visits each fixed-pool chunk exactly once.  Exact divisibility
    is required so no walkers disappear behind a partial tail.

    Returns:
        The contiguous chunk index selected for ``iteration``.

    Raises:
        ValueError: If the iteration is negative or the pool has a partial tail.
    """
    if int(iteration) < 0:
        raise ValueError("Training iteration must be nonnegative.")
    chunk_size = min(max(1, int(requested_size)), max(1, int(pool_size)))
    if chunk_size >= int(pool_size):
        return 0
    if int(pool_size) % chunk_size != 0:
        raise ValueError(
            f"Source pool size {pool_size} must be divisible by training "
            f"chunk size {chunk_size}."
        )
    chunk_count = int(pool_size) // chunk_size
    epoch, position = divmod(int(iteration), chunk_count)
    seed_sequence = np.random.SeedSequence(
        (
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            epoch & 0xFFFFFFFF,
            (epoch >> 32) & 0xFFFFFFFF,
        )
    )
    order = np.random.default_rng(seed_sequence).permutation(chunk_count)
    return int(order[position])


def _shuffled_batched_data_chunk(
    pool: BatchedData,
    requested_size: int,
    iteration: int,
    *,
    seed: int,
) -> BatchedData:
    chunk_index = _shuffled_batched_data_chunk_index(
        pool.batch_size,
        requested_size,
        iteration,
        seed=seed,
    )
    return _indexed_batched_data_chunk(pool, requested_size, chunk_index)


def _batched_data_chunks(pool: BatchedData, requested_size: int):
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    start = 0
    while start < pool.batch_size:
        size = min(chunk_size, pool.batch_size - start)
        yield _slice_batched_data(pool, start, size)
        start += size


@jax.jit
def _lit_stats_from_source_sums(
    sums: LITSourceSums,
    source_norm,
    omega,
    eta,
):
    """Convert held-out sums with one persistent fused executable.

    Returns:
        Standard LIT statistics for the accumulated source moments.
    """
    return stats_from_source_sums(
        sums,
        source_norm=source_norm,
        omega=omega,
        eta=eta,
    )


@jax.jit
def _add_source_sums(left, right):
    """Compatibility wrapper for the stable Chan/source-scale merge.

    Returns:
        The merged source-sampled accumulator.
    """
    return merge_source_sums(left, right)


def _signed_lit_jackknife_pseudovalues(
    full_stats: LITStats,
    block_sums: tuple[LITSourceSums, ...],
    *,
    source_norm: float,
    omega: float,
    eta: float,
) -> np.ndarray:
    """Return delete-one-block pseudovalues for the nonlinear LIT estimator.

    The signed transform is a ratio of accumulated moments, so treating each
    evaluation chunk as an independent transform and averaging those values is
    biased.  We instead merge all chunks except one, recompute the estimator,
    and form standard delete-one-block jackknife pseudovalues.  Prefix/suffix
    Chan merges keep this linear in the number of chunks.

    Raises:
        RuntimeError: If a defensive leave-one-out merge is unexpectedly empty.
    """
    block_count = len(block_sums)
    full_value = float(jax.device_get(full_stats.signed_lit))
    if block_count < 2:
        return np.asarray([full_value], dtype=np.float64)

    prefix: list[LITSourceSums | None] = [None]
    for block in block_sums:
        previous = prefix[-1]
        prefix.append(block if previous is None else _add_source_sums(previous, block))

    suffix: list[LITSourceSums | None] = [None] * (block_count + 1)
    for block_index in range(block_count - 1, -1, -1):
        following = suffix[block_index + 1]
        block = block_sums[block_index]
        suffix[block_index] = (
            block if following is None else _add_source_sums(block, following)
        )

    leave_one_out = np.empty(block_count, dtype=np.float64)
    for block_index in range(block_count):
        left = prefix[block_index]
        right = suffix[block_index + 1]
        if left is None:
            excluded = right
        elif right is None:
            excluded = left
        else:
            excluded = _add_source_sums(left, right)
        if excluded is None:  # Defensive; block_count >= 2 makes this unreachable.
            msg = "Jackknife exclusion produced an empty evaluation pool."
            raise RuntimeError(msg)
        excluded_stats = _lit_stats_from_source_sums(
            excluded,
            jnp.asarray(source_norm),
            jnp.asarray(omega),
            jnp.asarray(eta),
        )
        leave_one_out[block_index] = float(jax.device_get(excluded_stats.signed_lit))

    return block_count * full_value - (block_count - 1) * leave_one_out


def _merge_source_sums_across_devices(
    local_sums: LITSourceSums,
    *,
    axis_name: str = parallel_jax.BATCH_AXIS_NAME,
) -> LITSourceSums:
    """Compatibility wrapper for the stable data-parallel Chan merge.

    Returns:
        The globally merged source-sampled accumulator on every device.
    """
    return merge_source_sums_across_devices(
        local_sums,
        axis_name=axis_name,
    )


def _save_batched_pool(
    path: UPath,
    pool: BatchedData,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "fields_with_batch": np.asarray(list(pool.fields_with_batch), dtype=str),
    }
    if metadata is not None:
        for key, value in metadata.items():
            encoded = np.asarray(value)
            if encoded.ndim != 0 or encoded.dtype.hasobject:
                msg = f"source pool metadata {key!r} must be a scalar value"
                raise TypeError(msg)
            payload[f"metadata_{key}"] = encoded
    for field_name in pool.fields_with_batch:
        payload[field_name] = np.asarray(jax.device_get(getattr(pool.data, field_name)))
    _save_npz(path, **payload)


def _load_batched_pool(
    path: UPath,
    reference: BatchedData,
    *,
    metadata: Mapping[str, object] | None = None,
) -> BatchedData:
    with path.open("rb") as f_in, np.load(f_in, allow_pickle=False) as npf:
        if metadata is not None:
            _validate_pool_metadata(npf, metadata)
        fields = tuple(str(field) for field in npf["fields_with_batch"].tolist())
        if fields != tuple(reference.fields_with_batch):
            msg = (
                "source pool batched fields do not match current data: "
                f"{fields} != {tuple(reference.fields_with_batch)}"
            )
            raise ValueError(msg)
        updates = {}
        for field_name in fields:
            if field_name not in npf:
                msg = f"source pool is missing field {field_name!r}"
                raise KeyError(msg)
            value = jnp.asarray(npf[field_name])
            reference_value = getattr(reference.data, field_name)
            if value.shape[1:] != reference_value.shape[1:]:
                msg = (
                    f"source pool field {field_name!r} has incompatible shape "
                    f"{value.shape}; expected trailing {reference_value.shape[1:]}"
                )
                raise ValueError(msg)
            updates[field_name] = value
    return reference.__class__(
        data=reference.data.merge(updates),
        fields_with_batch=fields,
    )


def _validate_pool_metadata(npf, metadata: Mapping[str, object]) -> None:
    for key, expected in metadata.items():
        npz_key = f"metadata_{key}"
        if npz_key not in npf:
            msg = f"source pool is missing metadata {key!r}"
            raise ValueError(msg)
        encoded = np.asarray(npf[npz_key])
        if encoded.ndim != 0:
            msg = f"source pool metadata {key!r} must be scalar"
            raise ValueError(msg)
        actual: Any = encoded.item()
        matches: bool
        if isinstance(expected, str):
            matches = isinstance(actual, str) and actual == expected
        else:
            numeric_expected: Any = expected
            try:
                matches = bool(
                    np.isclose(
                        float(actual),
                        float(numeric_expected),
                        rtol=1e-8,
                        atol=1e-10,
                    )
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            msg = f"source pool metadata {key!r} mismatch: {actual!r} != {expected!r}"
            raise ValueError(msg)


def _require_pool_walker_count(
    pool: BatchedData,
    *,
    expected_walkers: int,
    split: str,
) -> None:
    actual_walkers = int(pool.batch_size)
    if actual_walkers != int(expected_walkers):
        msg = (
            f"{split} source pool has {actual_walkers} walkers; expected exactly "
            f"{int(expected_walkers)} from workflow.batch_size * configured batches"
        )
        raise ValueError(msg)


def _copy_matching_parameters(target, source):
    if not jax.tree_util.tree_leaves(source):
        return target
    target_mut = unfreeze(target)
    source_mut = unfreeze(source)
    return freeze(_copy_matching_mapping(target_mut, source_mut))


def _copy_matching_mapping(target, source):
    if isinstance(target, dict) and isinstance(source, dict):
        return {
            key: _copy_matching_mapping(value, source[key]) if key in source else value
            for key, value in target.items()
        }
    if (
        hasattr(target, "shape")
        and hasattr(source, "shape")
        and target.shape == source.shape
    ):
        return jnp.asarray(source, dtype=target.dtype)
    return target
