# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp

from jaqmc.app.solid.config import SolidConfig
from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.geometry.pbc import wrap_positions
from jaqmc.utils.atomic import distribute_spins, initialize_electrons_gaussian
from jaqmc.utils.supercell import get_supercell_copies


class SolidData(Data):
    """Container for solid state calculation data.

    Attributes:
        electrons: Electron positions for one walker. Built-in solid workflows
            usually use ``(n_elec, 3)`` per walker and ``(batch, n_elec, 3)``
            when this field is batched.
        atoms: Simulation-cell atom positions shared across walkers, with shape
            ``(n_atoms, 3)``.
        primitive_atoms: Primitive-cell atom positions shared across walkers.
        charges: Atomic charges shared across walkers.
    """

    electrons: jnp.ndarray
    atoms: jnp.ndarray
    primitive_atoms: jnp.ndarray
    charges: jnp.ndarray


def data_init(config: SolidConfig, size: int, rngs: PRNGKey) -> BatchedData[SolidData]:
    """Initializes a batch of solid data.

    Args:
        config: The solid configuration.
        size: The batch size (number of walkers).
        rngs: Random number generator key.

    Returns:
        The initialized batched data containing electron positions, lattice info,
        and k-points.
    """
    atoms = config.atoms
    lattice_vectors = jnp.asarray(config.supercell_lattice)

    atom_coords = jnp.stack([atom.coords_array for atom in config.atoms])
    atom_charges = jnp.array([atom.charge for atom in config.atoms])

    S = jnp.array(config.supercell_matrix)
    Rpts = get_supercell_copies(jnp.array(config.lattice_vectors), S)
    super_atom_coords = Rpts[:, None, :] + atom_coords[None, :, :]
    super_atom_coords = super_atom_coords.reshape(-1, 3)
    super_atom_charges = jnp.tile(atom_charges, len(Rpts))

    prim_atom_coords = jnp.stack([atom.coords_array for atom in atoms], axis=0)

    net_charge_prim = sum(atom.charge for atom in config.atoms) - sum(
        config.electron_spins
    )
    if net_charge_prim != 0 and config.fixed_spins_per_atom is None:
        raise NotImplementedError(
            "No initialization policy yet exists for charged solid."
        )

    rngs_position, rngs_spins = jax.random.split(rngs)
    fixed_spins_per_atom = config.fixed_spins_per_atom or distribute_spins(
        rngs_spins, atoms, config.electron_spins
    )
    electron_positions = initialize_electrons_gaussian(
        rngs_position,
        super_atom_coords,
        fixed_spins_per_atom * config.scale,
        size,
        config.electron_init_width,
    )

    electron_positions = wrap_positions(electron_positions, jnp.array(lattice_vectors))

    return BatchedData(
        data=SolidData(
            electrons=electron_positions,
            atoms=super_atom_coords,
            primitive_atoms=prim_atom_coords,
            charges=super_atom_charges,
        ),
        fields_with_batch=["electrons"],
    )
