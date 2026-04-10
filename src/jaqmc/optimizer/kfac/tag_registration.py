# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
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


def _flax_dense(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array], axes: int
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
            y = jax.lax.dot_general(x, w, (dimensions_in, ((), ())))
            # This reshape is required to match flax dense layer
            return y + b.reshape((1, *b.shape))
        case [w]:
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
        example_args=[np.zeros(x_shape), [np.zeros(s) for s in p_shapes]],
    )


def _make_scalar_scale_pattern() -> kfac_jax.tag_graph_matcher.GraphPattern:
    """Scale pattern for scalar (rank-0) parameters.

    KFAC-JAX's built-in scale patterns only cover rank >= 1.  This pattern
    matches ``x * scale`` where both operands are scalars, enabling KFAC to
    recognise scalar variational parameters (e.g. a single exponent alpha).

    Returns:
        A ``GraphPattern`` matching scalar multiplication.
    """
    # noinspection PyProtectedMember
    return kfac_jax.tag_graph_matcher.GraphPattern(
        name="scalar_scale",
        tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
        compute_func=functools.partial(
            kfac_jax.tag_graph_matcher._scale_and_shift,
            has_scale=True,
            has_shift=False,
        ),
        parameters_extractor_func=kfac_jax.tag_graph_matcher._scale_and_shift_parameter_extractor,
        example_args=[np.zeros(()), [np.zeros(())]],
    )


def make_graph_patterns():
    return (
        _make_scalar_scale_pattern(),
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
