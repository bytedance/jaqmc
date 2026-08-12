# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.geometry.pbc import wrap_positions

from .config import MoireConfig


class MoireData(Data):
    r"""Container for moire calculation data.

    Attributes:
        positions: Electron positions for one walker. Moire workflows use
            ``(n_elec, 2)`` per walker and ``(batch, n_elec, 2)`` when this
            field is batched.
        spin_coords: Continuous layer-pseudospin angles for one walker, with
            shape ``(n_elec,)`` per walker and ``(batch, n_elec)`` when batched.
            Each angle :math:`s\in[0,2\pi)` is the continuous-spin
            representation of the discrete top/bottom **layer** index, not the
            physical electron spin (which is fixed per channel by ``nspins``).
    """

    positions: jnp.ndarray
    spin_coords: jnp.ndarray


def data_init(
    config: MoireConfig,
    size: int,
    rngs: PRNGKey,
) -> BatchedData[MoireData]:
    """Initializes a batch of moire data.

    Args:
        config: The moire configuration.
        size: The batch size (number of walkers).
        rngs: Random number generator key.

    Returns:
        The initialized batched data containing electron positions and spin
        angles.
    """
    lattice = jnp.asarray(config.supercell_lattice)
    n_elec = config.nelec
    key_pos, key_spin = jax.random.split(rngs)
    # Initialize electron positions uniformly in the simulation cell.
    fractional = jax.random.uniform(key_pos, (size, n_elec, 2))
    positions = wrap_positions(fractional @ lattice, lattice)
    # Initialize continuous layer-pseudospin angles uniformly on [0, 2pi).
    spin_coords = jax.random.uniform(key_spin, (size, n_elec), maxval=2.0 * jnp.pi)

    return BatchedData(
        data=MoireData(
            positions=positions,
            spin_coords=spin_coords,
        ),
        fields_with_batch=["positions", "spin_coords"],
    )
