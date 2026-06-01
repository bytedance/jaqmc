# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

import functools
import itertools
from collections.abc import Sequence

import jax
import kfac_jax
import numpy as np
from jax import numpy as jnp


def _flax_dense(
    x: kfac_jax.utils.Array,
    params: Sequence[kfac_jax.utils.Array],
    axes: int,
) -> kfac_jax.utils.Array:
    """Example of a dense layer function.

    This assumes the contraction happens over the last `axes` dimensions of `x` and the
    first `axes` dimensions of `params`.
    We use `dot_general` instead of `tenserdot` which is used in original kfac_jax code
    so that operations in flax can be recognized by KFAC.

    Returns:
        Example outputs.

    Raises:
        ValueError: Params neither [w, b] or [w].
    """
    dimensions_in = (
        tuple(x.ndim - 1 - i for i in reversed(range(axes))),
        tuple(range(axes)),
    )
    match params:
        case [w, b]:
            w = w.astype(x.dtype)
            b = b.astype(x.dtype)
            y = jax.lax.dot_general(x, w, (dimensions_in, ((), ())))
            # This reshape is required to match flax dense layer
            return y + b.reshape((1, *b.shape))
        case [w]:
            w = w.astype(x.dtype)
            return jax.lax.dot_general(x, w, (dimensions_in, ((), ())))
        case _:
            raise ValueError("Unsupported parameters list")


def _make_flax_dense_pattern(
    with_bias: bool,
    num_repeated_axes: int,
    num_in_dims: int,
    num_out_dims: int,
) -> kfac_jax.tag_graph_matcher.GraphPattern:
    """Returns a pattern for a dense or repeated dense layer."""
    batch_dim = (2,)
    repeating_dims = tuple(itertools.repeat(7, num_repeated_axes))

    out_dims = tuple(i + 2 for i in range(num_out_dims))
    in_dims = tuple(i + 2 for i in range(num_in_dims))
    x_shape = batch_dim + repeating_dims + in_dims
    weight_shape = in_dims + out_dims
    p_shapes = [weight_shape, out_dims] if with_bias else [weight_shape]

    name = "flax_dense_with_bias" if with_bias else "flax_dense_no_bias"

    if num_repeated_axes > 0:
        name = f"repeated[{num_repeated_axes}]_{name}"
        variant = "repeated_dense"
    else:
        variant = "dense"

    return kfac_jax.tag_graph_matcher.GraphPattern(
        name=name,
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=functools.partial(_flax_dense, axes=num_in_dims),
        parameters_extractor_func=functools.partial(
            kfac_jax.tag_graph_matcher._dense_parameter_extractor, variant=variant
        ),
        example_args=[
            np.zeros(x_shape),
            [np.zeros(s, dtype=np.float32) for s in p_shapes],
        ],
    )


def _scale_and_shift(
    x: kfac_jax.utils.Array,
    params: Sequence[kfac_jax.utils.Array],
    has_scale: bool,
    has_shift: bool,
) -> kfac_jax.utils.Array:
    if has_scale and has_shift:
        scale, shift = params
        return x * scale.astype(x.dtype) + shift.astype(x.dtype)
    elif has_scale:
        [scale] = params
        return x * scale.astype(x.dtype)
    elif has_shift:
        [shift] = params
        return x + shift.astype(x.dtype)
    else:
        raise ValueError("You must have either `has_scale` or `has_shift` set to True.")


def _make_scalar_scale_pattern(
    x_ndim: int, has_scale: bool, has_shift: bool
) -> kfac_jax.tag_graph_matcher.GraphPattern:
    """Build a scale/shift pattern that handles mixed parameter dtypes.

    Returns:
        A KFAC graph pattern for the requested broadcast rank and operations.

    Raises:
        ValueError: If neither scaling nor shifting is requested.
    """
    assert x_ndim >= 0
    assert has_scale or has_shift

    x_shape = [i + 2 for i in range(x_ndim)]
    p_shapes = (
        [x_shape[-1:], x_shape[-1:]] if (has_scale and has_shift) else [x_shape[-1:]]
    )

    if has_scale and has_shift:
        name = f"scale_and_shift_broadcast_{x_ndim}"
    elif has_scale:
        name = f"scale_only_broadcast_{x_ndim}"
    elif has_shift:
        name = f"shift_only_broadcast_{x_ndim}"
    else:
        raise ValueError("Unreachable.")

    return kfac_jax.tag_graph_matcher.GraphPattern(
        name=name,
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=functools.partial(
            _scale_and_shift, has_scale=has_scale, has_shift=has_shift
        ),
        parameters_extractor_func=kfac_jax.tag_graph_matcher._scale_and_shift_parameter_extractor,
        example_args=[
            np.zeros(x_shape),
            [np.zeros(s, dtype=np.float32) for s in p_shapes],
        ],
    )
    # noinspection PyProtectedMember
    return kfac_jax.tag_graph_matcher.GraphPattern(
        name="scalar_scale",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=functools.partial(
            _scale_and_shift, has_scale=True, has_shift=False
        ),
        parameters_extractor_func=kfac_jax.tag_graph_matcher._scale_and_shift_parameter_extractor,
        example_args=[
            np.zeros(()),
            [np.zeros((), dtype=np.float32)],
        ],
    )


def _flax_vmapped_normalization(
    inputs: Sequence[kfac_jax.utils.Array],
    params: Sequence[kfac_jax.utils.Array],
    has_shift: bool,
) -> kfac_jax.utils.Array:
    """Affine tail emitted by Flax LayerNorm under an outer `jax.vmap`.

    Returns:
        LayerNorm output with scale and optional shift broadcast to the
        vmapped activation shape.
    """
    [x, rsqrt_var] = inputs
    scale = params[0]
    scale = scale.reshape((1, *scale.shape))
    scale = scale.astype(x.dtype)
    scale = jnp.broadcast_to(scale, (1,) * (x.ndim - scale.ndim) + scale.shape)
    y = x * (rsqrt_var * scale)
    if not has_shift:
        return y

    shift = params[1]
    shift = shift.reshape((1, *shift.shape))
    shift = shift.astype(x.dtype)
    shift = jnp.broadcast_to(shift, (1,) * (x.ndim - shift.ndim) + shift.shape)
    return y + shift


def _make_flax_vmapped_normalization_pattern(
    broadcast_ndim: int,
    has_shift: bool,
    p_dim: int = 13,
) -> kfac_jax.tag_graph_matcher.GraphPattern:
    """Pattern for Flax LayerNorm traced through an outer `jax.vmap`.

    Flax lowers vmapped LayerNorm with the normalization inverse shaped like
    `[..., 1]` instead of fully broadcasting it to `[..., feature_dim]`.
    KFAC-JAX's built-in normalization patterns only cover the fully-broadcasted
    form, so the affine scale/bias parameters fall back to `generic`.

    Returns:
        A ``GraphPattern`` matching the vmapped LayerNorm affine tail.
    """
    x_shape = [i + 2 for i in range(broadcast_ndim)] + [p_dim]
    rsqrt_shape = [i + 2 for i in range(broadcast_ndim)] + [1]
    example_params = [np.zeros([p_dim], dtype=np.float32)]
    if has_shift:
        example_params.append(np.zeros([p_dim], dtype=np.float32))

    return kfac_jax.tag_graph_matcher.GraphPattern(
        name=f"flax_vmapped_normalization_broadcast_{broadcast_ndim}",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=functools.partial(
            _flax_vmapped_normalization,
            has_shift=has_shift,
        ),
        parameters_extractor_func=kfac_jax.tag_graph_matcher._scale_and_shift_parameter_extractor,
        example_args=[[np.zeros(x_shape), np.zeros(rsqrt_shape)], example_params],
        in_values_preprocessor=kfac_jax.tag_graph_matcher._normalization_haiku_preprocessor,
    )


def make_graph_patterns():
    return (
        *tuple(
            _make_scalar_scale_pattern(x_ndim, has_scale, has_shift)
            for x_ndim in (0, 1, 2)
            for has_scale, has_shift in [(True, True), (True, False), (False, True)]
        ),
        *tuple(
            _make_flax_vmapped_normalization_pattern(
                broadcast_ndim=broadcast_ndim,
                has_shift=has_shift,
            )
            for broadcast_ndim, has_shift in itertools.product(
                range(1, 3), (False, True)
            )
        ),
        *tuple(
            _make_flax_dense_pattern(
                with_bias=with_bias,
                num_repeated_axes=repeat,
                num_in_dims=n_ins,
                num_out_dims=n_outs,
            )
            for with_bias, repeat, n_ins, n_outs in itertools.product(
                (True, False), range(3), range(1, 3), range(1, 3)
            )
        ),
        *kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS,
    )
