# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey

from .atom import Atom
from .atomic_system import AtomInitialization


def _atom_initial_spin_config(atom: Atom, initialization: AtomInitialization):
    if not hasattr(atom, "charge"):
        raise ValueError(
            f"Atom charge for {atom.symbol!r} is unresolved. Resolve the atom "
            "within an AtomicSystemConfig or pass charge explicitly."
        )
    spin_imbalance = initialization.spin_imbalance
    local_charge = initialization.local_charge

    if spin_imbalance is not None:
        nelec = atom.charge + local_charge
        spins = ((nelec + spin_imbalance) // 2, (nelec - spin_imbalance) // 2)
    elif local_charge > atom.spin_config[0]:
        spins = (0, sum(atom.spin_config) - local_charge)
    else:
        spins = (atom.spin_config[0] - local_charge, atom.spin_config[1])
    if any(count < 0 for count in spins):
        raise ValueError(
            "Atom-local initialization must satisfy non-negative per-spin occupancies; "
            f"got symbol={atom.symbol!r}, spin_imbalance={spin_imbalance}, "
            f"local_charge={local_charge}, spins={spins}."
        )
    return spins


def distribute_spins(
    rngs: PRNGKey,
    atoms: list[Atom],
    per_atom_init: list[AtomInitialization],
    total_spins: tuple[int, int],
) -> list[tuple[int, int]]:
    """Resolve a full per-atom initialization from partial user overrides.

    Explicit atom initialization hints are first resolved into exact local
    ``(n_alpha, n_beta)`` occupancies and then preserved exactly. Unspecified
    atoms start from their chemical default ``atom.spin_config``, then absorb
    any remaining channel-by-channel difference required to match
    ``total_spins``.

    Args:
        rngs: Random number generator key.
        atoms: List of atoms defining the system.
        per_atom_init: Per-atom initialization hints aligned with ``atoms``.
        total_spins: Target system ``(n_alpha, n_beta)``.

    Returns:
        Full per-atom initialization spins.

    Raises:
        ValueError: If the explicit overrides over-constrain the target system
            and the unspecified atoms cannot absorb the remaining difference.
    """
    local_charge_sum = sum(init.local_charge for init in per_atom_init)
    if not local_charge_sum == 0:
        raise ValueError(
            f"All per-atom charge offsets must sum to zero. Got {local_charge_sum}."
        )
    spins_per_atom = list(map(_atom_initial_spin_config, atoms, per_atom_init))
    fixed_per_atom = [init.spin_imbalance is not None for init in per_atom_init]
    explicit_spins = [
        spin for spin, fixed in zip(spins_per_atom, fixed_per_atom) if fixed
    ]
    specified_spins = (
        tuple(sum(channel) for channel in zip(*explicit_spins))
        if explicit_spins
        else (0, 0)
    )
    total_electrons = sum(sum(spins) for spins in spins_per_atom)
    if sum(total_spins) != total_electrons:
        raise ValueError(
            "After applying the explicit initialization hints, the system has "
            f"{total_electrons} electrons but total_spins={total_spins} "
            f"requires {sum(total_spins)}."
        )

    if all(fixed_per_atom):
        if specified_spins != total_spins:
            raise ValueError(
                f"Expected total spins {total_spins}. Got {specified_spins} specified."
            )
        return explicit_spins

    mutable_indices = [i for i, fixed in enumerate(fixed_per_atom) if not fixed]
    current_alpha = sum(spins[0] for spins in spins_per_atom)
    target_alpha = total_spins[0]
    while current_alpha != target_alpha:
        move_alpha_up = target_alpha > current_alpha
        candidates = [
            i
            for i in mutable_indices
            if spins_per_atom[i][1 if move_alpha_up else 0] > 0
        ]
        if not candidates:
            raise ValueError(
                "The system needs remaining spins that cannot be assigned to the "
                f"mutable atoms. Current per-atom spins: {spins_per_atom}, "
                f"target total: {total_spins}."
            )
        rngs, subkey = jax.random.split(rngs)
        i = int(jax.random.randint(subkey, (), 0, len(candidates)))
        nalpha, nbeta = spins_per_atom[candidates[i]]
        if move_alpha_up:
            spins_per_atom[candidates[i]] = (nalpha + 1, nbeta - 1)
            current_alpha += 1
        else:
            spins_per_atom[candidates[i]] = (nalpha - 1, nbeta + 1)
            current_alpha -= 1
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
        spins_per_atom: List of ``(n_alpha, n_beta)`` tuples for each atom, dictating
            how many electrons are initialized around each nucleus.
        batch_size: Number of walkers.
        init_width: Standard deviation (scale) of the Gaussian noise added to
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
