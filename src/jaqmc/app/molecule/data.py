# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp

from jaqmc.app.molecule.config import MoleculeConfig
from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.utils.atomic import distribute_spins, initialize_electrons_gaussian


class MoleculeData(Data):
    """Container for molecular simulation data.

    Attributes:
        electrons: Electron coordinates for one walker. Built-in molecule
            workflows usually use ``(n_elec, 3)`` per walker and
            ``(batch, n_elec, 3)`` when this field is batched.
        atoms: Atomic coordinates shared across walkers, with shape
            ``(n_atoms, 3)``.
        charges: Atomic charges shared across walkers, with shape
            ``(n_atoms,)``.
    """

    electrons: jnp.ndarray
    atoms: jnp.ndarray
    charges: jnp.ndarray


def data_init(
    config: MoleculeConfig, size: int, rngs: PRNGKey
) -> BatchedData[MoleculeData]:
    electron_spins = config.electron_spins
    fixed_spins_per_atom = config.fixed_spins_per_atom

    net_charge = sum(atom.charge for atom in config.atoms) - sum(electron_spins)
    if net_charge != 0 and fixed_spins_per_atom is None:
        raise NotImplementedError(
            "No initialization policy yet exists for charged molecules."
        )

    rngs_position, rngs_spins = jax.random.split(rngs)
    electron_positions = initialize_electrons_gaussian(
        rngs_position,
        jnp.stack([atom.coords_array for atom in config.atoms], axis=0),
        fixed_spins_per_atom
        or distribute_spins(rngs_spins, config.atoms, electron_spins),
        size,
        config.electron_init_width,
    )

    return BatchedData(
        data=MoleculeData(
            electrons=electron_positions,
            atoms=jnp.stack([atom.coords_array for atom in config.atoms], axis=0),
            charges=jnp.array([atom.charge for atom in config.atoms]),
        ),
        fields_with_batch=["electrons"],
    )
