# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import lax
from jax import numpy as jnp

from jaqmc.array_types import PyTree


def chunked_vmap(
    fun, in_axes: PyTree = 0, out_axes: PyTree = 0, chunk_size: int | None = None
):
    """Return a ``vmap`` wrapper that evaluates mapped inputs in chunks.

    The wrapped function is intended to match ``jax.vmap`` for the supported
    ``in_axes``/``out_axes`` signatures while reducing peak memory use. When
    ``chunk_size`` is ``None`` or at least the mapped batch size, this delegates
    directly to ``jax.vmap``.

    Args:
        fun: Function to vectorize.
        in_axes: Axis specification for mapped inputs, following ``jax.vmap``.
            This may be a prefix tree.
        out_axes: Axis specification for mapped outputs, following ``jax.vmap``.
            Chunk outputs are collected by ``lax.scan`` and then restored to
            this layout. Literal ``None`` output leaves are preserved as is.
        chunk_size: Positive number of mapped inputs to evaluate per chunk. If
            ``None``, or at least the mapped batch size, evaluation delegates
            to plain ``jax.vmap``.

    Returns:
        A wrapped function with the same signature as ``fun``. When called, the
        wrapper requires at least one mapped input leaf and all mapped input
        leaves must share one batch size.
    """

    def wrapped(*args):
        return _chunked_vmap_wrapped(fun, in_axes, out_axes, chunk_size, *args)

    return wrapped


def _chunked_vmap_wrapped(fun, in_axes, out_axes, chunk_size, *args):
    """Evaluate ``fun`` with chunked ``vmap`` semantics.

    This function expands prefix ``in_axes`` before slicing inputs, evaluates
    full chunks through ``lax.scan``, evaluates one remainder chunk, and then
    restores each output leaf to the user-requested ``out_axes`` layout.

    Returns:
        The output pytree produced by ``fun`` with the same mapped-axis layout
        as ``jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)(*args)``.
    """
    # Slicing needs a concrete axis spec for every input leaf. Public vmap
    # accepts prefix trees, so expand that prefix form before chunking.
    full_in_axes = _tree_broadcast(in_axes, args, is_leaf=_none_is_leaf)
    # Mapped output leaves are produced on axis 0 inside each chunk. Leaves
    # with out_axes=None stay unmapped, as in plain jax.vmap.
    chunk_out_axes = jax.tree.map(lambda _: 0, out_axes)
    vmapped_chunk = jax.vmap(fun, in_axes=full_in_axes, out_axes=chunk_out_axes)

    mapped_axis_sizes = _mapped_axis_sizes(args, full_in_axes)
    batch_size = _validate_batch_sizes(mapped_axis_sizes)
    if chunk_size is None or chunk_size >= batch_size:
        return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)(*args)

    num_chunks = batch_size // chunk_size
    remainder_size = batch_size % chunk_size

    def scan_body(_, chunk_idx):
        start = chunk_idx * chunk_size
        chunk_args = _take_chunk_from_args(args, full_in_axes, start, chunk_size)
        return None, vmapped_chunk(*chunk_args)

    _, chunk_outputs = lax.scan(scan_body, None, jnp.arange(num_chunks))

    # Always run the remainder path, even when remainder_size is 0. That keeps
    # the reconstruction logic uniform and gives the output tree used to
    # broadcast out_axes below.
    remainder_start = num_chunks * chunk_size
    remainder_args = _take_chunk_from_args(
        args, full_in_axes, remainder_start, remainder_size
    )
    remainder_outputs = vmapped_chunk(*remainder_args)
    full_out_axes = _tree_broadcast(out_axes, remainder_outputs, is_leaf=_none_is_leaf)
    return jax.tree.map(
        lambda chunked_output, remainder_output, out_axis: _restore_vmap_output_leaf(
            chunked_output,
            remainder_output,
            out_axis,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        ),
        chunk_outputs,
        remainder_outputs,
        full_out_axes,
        is_leaf=_none_is_leaf,
    )


def _none_is_leaf(x):
    """Treat literal ``None`` as a pytree leaf.

    This lets axis broadcasting and output reconstruction preserve ``None`` as
    data instead of treating it as an empty pytree.

    Returns:
        ``True`` when ``x`` is ``None``.
    """
    return x is None


def _mapped_axis_sizes(args, full_in_axes):
    """Return only mapped-axis batch sizes from ``args``.

    For each input leaf, this computes ``x.shape[ax]`` when ``ax`` is mapped
    and uses ``None`` when ``ax`` is ``None`` (unmapped). The final
    ``jax.tree.leaves`` call drops those ``None`` placeholders by default, so
    the returned list contains sizes for mapped leaves only.

    Example:
        Axis sizes ``[8, None, 4]`` become ``[8, 4]``.

    Returns:
        A list of mapped-axis batch sizes, one entry per mapped input leaf.
    """
    # Build per-leaf mapped sizes, using None as a placeholder for unmapped
    # leaves. jax.tree.leaves then drops None entries by default.
    return jax.tree.leaves(
        jax.tree.map(
            lambda x, ax: x.shape[ax] if ax is not None else None,
            args,
            full_in_axes,
            is_leaf=_none_is_leaf,
        )
    )


def _validate_batch_sizes(axes_sizes):
    """Validate that all mapped input leaves share one batch size.

    Returns:
        The common mapped batch size.

    Raises:
        ValueError: If no input leaf is mapped or mapped leaf sizes disagree.
    """
    if not axes_sizes:
        raise ValueError("chunked_vmap requires at least one mapped axis")
    total_size = axes_sizes[0]
    for size in axes_sizes[1:]:
        if size != total_size:
            raise ValueError("Inconsistent batch sizes")
    return total_size


def _take_chunk_from_args(args, full_in_axes, start, size):
    """Slice mapped input leaves to one chunk.

    Leaves whose axis spec is ``None`` are not mapped and are passed through
    unchanged to every chunk.

    Returns:
        A pytree matching ``args`` where mapped leaves contain the requested
        chunk and unmapped leaves are unchanged.
    """
    return jax.tree.map(
        lambda x, ax: (
            lax.dynamic_slice_in_dim(x, start, size, ax) if ax is not None else x
        ),
        args,
        full_in_axes,
        is_leaf=_none_is_leaf,
    )


def _restore_vmap_output_leaf(
    chunked_output, remainder_output, out_axis, *, num_chunks, chunk_size
):
    """Restore one output leaf to the layout requested by ``out_axes``.

    Full chunk outputs arrive from ``lax.scan`` with shape
    ``[num_chunks, chunk_size, ...]`` for mapped leaves. This helper flattens
    those scan/chunk axes, appends the remainder output, and moves the mapped
    axis to ``out_axis``.

    Returns:
        One reconstructed output leaf.
    """
    # A literal None output leaf is data, not a chunked array. Plain jax.vmap
    # preserves it, so the chunked path must preserve it too.
    if chunked_output is None:
        return None
    if out_axis is None:
        # For out_axes=None, vmapped_chunk returns an unmapped value for each
        # full chunk, and lax.scan stacks those values over chunks. All chunks
        # represent the same unmapped output, so return the first one just as
        # plain jax.vmap would return a single value.
        return chunked_output[0]

    # lax.scan stacks chunk results as [num_chunks, chunk_size, ...]. Plain vmap
    # has one mapped axis, so flatten the first two axes before applying out_axis.
    chunked_output = chunked_output.reshape(
        num_chunks * chunk_size, *chunked_output.shape[2:]
    )
    full_output = jnp.concatenate([chunked_output, remainder_output], axis=0)
    return jnp.moveaxis(full_output, 0, out_axis)


def _tree_broadcast(prefix_tree, full_tree, **kw):
    """Broadcast an axes prefix tree to match a full pytree.

    Modern JAX provides ``jax.tree.broadcast`` for the prefix-tree semantics
    accepted by ``jax.vmap``. The fallback only handles scalar integer axes and
    top-level ``None``; nested prefix trees must already match ``full_tree`` on
    older JAX versions.

    Returns:
        ``prefix_tree`` expanded to the structure of ``full_tree`` when
        broadcasting is available or supported by the fallback.
    """
    # Modern JAX supports the same prefix-tree broadcasting needed by vmap.
    if hasattr(jax.tree, "broadcast"):
        return jax.tree.broadcast(prefix_tree, full_tree, **kw)
    # Older JAX fallback
    from jax._src.tree_util import broadcast_prefix

    leaves = broadcast_prefix(prefix_tree, full_tree, **kw)
    return jax.tree.structure(full_tree).unflatten(leaves)
