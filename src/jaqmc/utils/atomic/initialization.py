# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.utils.atomic.atom import Atom


def distribute_spins(
    rngs: PRNGKey,
    atoms: list[Atom],
    total_spins: tuple[int, int],
) -> list[tuple[int, int]]:
    """Distributes total spins among atoms to match the target system spin.

    The function initializes spins from atomic defaults and iteratively
    adjusts the distribution (transferring spins between channels) until the
    global `total_spins` constraint is met.

    Args:
        rngs: Random number generator key.
        atoms: List of Atom objects defining the system.
        total_spins: A tuple (n_alpha, n_beta) representing the total number
            of electrons in the system.

    Returns:
        A list of tuples, where each tuple (n_alpha, n_beta) corresponds to the
        spin configuration of an atom in the `atoms` list.
    """
    if len(atoms) == 1:
        return [total_spins]

    # Initialize from hint or atomic defaults
    spins_per_atom = [atom.spin_config for atom in atoms]

    # Ensure the total number of electrons is consistent
    current_total = sum(sum(x) for x in spins_per_atom)
    target_total = sum(total_spins)
    assert current_total == target_total, (
        f"Total electrons mismatch: atoms sum to {current_total}, "
        f"but system requires {target_total}"
    )

    # Adjust spin distribution (alpha/beta balance) to match total_spins
    spins_per_atom_mutable = [list(x) for x in spins_per_atom]

    while tuple(sum(x) for x in zip(*spins_per_atom_mutable)) != total_spins:
        rngs, subkey = jax.random.split(rngs)
        i = int(jax.random.randint(subkey, (), 0, len(spins_per_atom_mutable)))
        nalpha, nbeta = spins_per_atom_mutable[i]
        if nalpha > 0:
            spins_per_atom_mutable[i] = [nalpha - 1, nbeta + 1]
    spins_per_atom = [tuple(x) for x in spins_per_atom_mutable]  # type: ignore
    return spins_per_atom


def initialize_electrons_gaussian(
    rng: PRNGKey,
    atom_coords: jnp.ndarray,
    spins_per_atom: list[tuple[int, int]],
    batch_size: int,
    init_width: float,
) -> jnp.ndarray:
    """Initializes electron positions with Gaussian distribution around atoms.

    Electrons are assigned to atoms based on the local spin configuration
    provided in `spins_per_atom`. The initial positions are centered on the
    atoms with added Gaussian noise.

    Args:
        rng: Random number generator key.
        atom_coords: Array of atom coordinates with shape (natoms, 3).
        spins_per_atom: List of (n_alpha, n_beta) tuples for each atom, dictating
            how many electrons are initialized around each nucleus.
        batch_size: Number of walkers.
        init_width: tandard deviation (scale) of the Gaussian noise added to
            the atomic centers.

    Returns:
        Electron positions array of shape (batch_size, nelec, 3).
    """
    # Assign each electron to an atom initially.
    electron_positions = jnp.concatenate(
        [
            jnp.tile(atom_coords[j], (spins_per_atom[j][i], 1))
            for i in range(2)
            for j in range(len(spins_per_atom))
        ]
    )

    # Create a batch of configurations with a Gaussian distribution about each atom.
    electron_positions += init_width * jax.random.normal(
        rng, shape=(batch_size, *electron_positions.shape)
    )

    return electron_positions
