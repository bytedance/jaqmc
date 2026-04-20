# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import chex
import jax
from jax import numpy as jnp

from jaqmc.utils.chunked_vmap import chunked_vmap


def test_chunked_vmap_matches_vmap_for_nested_axes():
    def f(inputs):
        return {
            "a": (inputs["x"][0] + 1, inputs["x"][1] - 1),
            "b": jnp.sum(inputs["z"]),
            "c": inputs["x"][0] ** 2 + inputs["y"],
        }

    inputs = {
        "x": (jnp.arange(21), jnp.arange(21)),
        "y": jnp.ones((3, 21)),
        "z": jnp.arange(5),
    }
    in_axes = ({"x": 0, "y": 1, "z": None},)
    out_axes = {"a": 0, "b": None, "c": 1}
    if not hasattr(jax.tree, "broadcast"):
        # no broadcast available, need to fully specify in_axes and out_axes
        in_axes[0]["x"] = (0, 0)
        out_axes["a"] = (0, 0)

    y_ref = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)(inputs)
    y = chunked_vmap(f, in_axes=in_axes, out_axes=out_axes, chunk_size=6)(inputs)

    chex.assert_trees_all_close(y, y_ref)


def test_chunked_vmap_handles_exact_chunk_partition():
    def f(x):
        return {"leading": x + 1, "nonleading": jnp.stack([x, x**2])}

    x = jnp.arange(21)
    out_axes = {"leading": 0, "nonleading": 1}

    y_ref = jax.vmap(f, out_axes=out_axes)(x)
    y = chunked_vmap(f, out_axes=out_axes, chunk_size=7)(x)

    chex.assert_trees_all_close(y, y_ref)


def test_chunked_vmap_preserves_none_output_leaf():
    def evaluate_local(params, data, prev_local_stats, state, rng):
        del params, prev_local_stats, rng
        return {"energy:kinetic": data * 2}, state

    params = {"w": jnp.array([1.0])}
    data = jnp.arange(10.0)
    prev_local_stats = {}
    state = None
    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    in_axes = (None, 0, 0, None, 0)

    y_ref = jax.vmap(evaluate_local, in_axes=in_axes)(
        params, data, prev_local_stats, state, rngs
    )
    y = chunked_vmap(evaluate_local, in_axes=in_axes, chunk_size=4)(
        params, data, prev_local_stats, state, rngs
    )

    chex.assert_trees_all_close(y, y_ref)
