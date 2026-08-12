# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Walker initialization for the homogeneous electron gas."""

import jax
from jax import numpy as jnp

from jaqmc.app.solid.data import SolidData
from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData

from .config import ElectronGasConfig


def data_init(
    config: ElectronGasConfig, size: int, rngs: PRNGKey
) -> BatchedData[SolidData]:
    """Initialize electrons uniformly in the simulation cell.

    Returns:
        Batched solid-compatible data with empty nuclear fields.
    """
    lattice = jnp.asarray(config.lattice)
    fractional = jax.random.uniform(rngs, (size, config.nelectrons, 3))
    electrons = fractional @ lattice

    return BatchedData(
        data=SolidData(
            electrons=electrons,
            # Keep separate empty buffers: workflow state is donated to compiled
            # steps, and JAX rejects donating the same buffer through two fields.
            atoms=jnp.empty((0, 3), dtype=electrons.dtype),
            primitive_atoms=jnp.empty((0, 3), dtype=electrons.dtype),
            charges=jnp.empty((0,), dtype=electrons.dtype),
        ),
        fields_with_batch=["electrons"],
    )
