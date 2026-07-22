# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Shared deterministic fixtures for sharding-focused Laplacian tests."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

STANDARD_MLP_INPUT = jnp.array([0.5, -0.3, 1.2, 0.8], dtype=jnp.float32)

MlpFn = Callable[[jnp.ndarray], jnp.ndarray]


def deterministic_two_layer_mlp() -> tuple[MlpFn, jnp.ndarray]:
    """Return a two-layer tanh MLP and its standard 4-element input."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    w1 = jax.random.normal(keys[0], (8, 4))
    b1 = jax.random.normal(keys[1], (8,))
    w2 = jax.random.normal(keys[2], (1, 8))
    b2 = jax.random.normal(keys[3], (1,))

    def mlp(x: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.tanh(w1 @ x + b1)
        return jnp.sum(jnp.tanh(w2 @ hidden + b2))

    return mlp, STANDARD_MLP_INPUT
