# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections.abc import Sequence

from jax import numpy as jnp


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
    """Returns the indices for splitting an array into separate partitions.

    Args:
        sizes: Size of each of N partitions. The dimension of the array along
            the relevant axis is assumed to be sum(sizes).

    Returns:
        Sequence of indices (length len(sizes)-1) at which an array should be split
        to give the desired partitions.
    """
    return list(itertools.accumulate(sizes))[:-1]


def split_nonempty_channels(x: jnp.ndarray, sizes: Sequence[int]) -> list[jnp.ndarray]:
    """Split an array into non-empty channels along its first axis.

    Args:
        x: Array to split. Its first axis must have length ``sum(sizes)``.
        sizes: Channel sizes along the first axis. Zero-sized channels are
            omitted from the result.

    Returns:
        Non-empty channel slices of ``x``. If zero or one channel is non-empty,
        the result is ``[x]``.

    Raises:
        ValueError: If ``x.shape[0]`` does not equal ``sum(sizes)``.
    """
    if x.shape[0] != sum(sizes):
        raise ValueError(
            f"Sizes {sizes} should match the first axis ({x.shape[0]}) of input array."
        )
    if sum(size > 0 for size in sizes) <= 1:
        return [x]
    return jnp.split(x, array_partitions([size for size in sizes if size > 0]))


def match_first_axis_of(x: jnp.ndarray, target: jnp.ndarray):
    """Reshape an array for broadcasting against a higher-rank target.

    Args:
        x: Array whose existing axes should be preserved.
        target: Array whose rank determines how many trailing singleton axes
            are appended to ``x``.

    Returns:
        ``x`` with enough trailing singleton axes to match ``target.ndim``.
    """
    return x[(...,) + (None,) * (target.ndim - x.ndim)]
