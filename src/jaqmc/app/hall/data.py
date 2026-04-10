# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Data structures for quantum Hall effect simulations."""

import jax
from jax import numpy as jnp

from jaqmc.app.hall.config import HallConfig
from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data


class HallData(Data):
    """Data container for quantum Hall simulations on the Haldane sphere.

    Attributes:
        electrons: Electron positions for one walker in spherical coordinates,
            with shape ``(n_elec, 2)`` where the last axis is ``(theta, phi)``.
            Built-in Hall workflows batch this field as
            ``(batch, n_elec, 2)``.
    """

    electrons: jnp.ndarray


def data_init(config: HallConfig, size: int, rngs: PRNGKey) -> BatchedData[HallData]:
    """Create uniform initial samples on the sphere.

    Args:
        config: Hall system configuration.
        size: Batch size (number of walkers).
        rngs: Random number generator key.

    Returns:
        Batched data with uniformly distributed electrons on the sphere.
    """
    nelec = sum(config.nspins)
    key1, key2 = jax.random.split(rngs)
    theta = jnp.arccos(jax.random.uniform(key1, (size, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (size, nelec), minval=-jnp.pi, maxval=jnp.pi)
    electrons = jnp.stack([theta, phi], axis=-1)
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )
