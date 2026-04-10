# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from jaqmc.estimator.ewald import EwaldSum
from jaqmc.utils import supercell


def build_supercell(lattice, atom_coords, atom_charges, S):
    """Builds supercell lattice, atoms, and charges from primitive cell info.

    Args:
        lattice: Primitive lattice vectors (3, 3).
        atom_coords: Atomic coordinates in primitive cell (natm, 3).
        atom_charges: Atomic charges (natm,).
        S: Supercell matrix (3, 3).

    Returns:
        Tuple of (supercell_lattice, supercell_atoms, supercell_charges, scale).
    """
    # Calculate supercell lattice
    supercell_lattice = jnp.dot(S, lattice)

    # Calculate scale (determinant of S)
    scale = round(float(jnp.linalg.det(S)))

    # Get translation vectors
    translations = supercell.get_supercell_copies(lattice, S)

    # Tile atoms
    supercell_atoms = (atom_coords[None, :, :] + translations[:, None, :]).reshape(
        -1, 3
    )

    # Tile charges
    supercell_charges = jnp.tile(atom_charges, scale)

    return supercell_lattice, supercell_atoms, supercell_charges


def get_ewald_energy(lattice, atom_coords, atom_charges, S, electron_coords):
    """Helper to setup supercell and compute Ewald energy.

    Returns:
        Total Ewald energy.
    """
    (
        supercell_lattice,
        supercell_atoms,
        supercell_charges,
    ) = build_supercell(lattice, atom_coords, atom_charges, S)

    nelec = len(electron_coords)

    # Concatenate all particles
    electron_charges = -jnp.ones(nelec)

    all_coords = jnp.concatenate([electron_coords, supercell_atoms], axis=0)
    all_charges = jnp.concatenate([electron_charges, supercell_charges], axis=0)

    # Initialize Ewald Sum
    ewald_solver = EwaldSum(supercell_lattice)

    # Compute energy
    etot = ewald_solver.energy(all_coords, all_charges)

    return etot.item()


def test_madelung_nacl_primitive():
    """Test Madelung constant for NaCl using Primitive Cell (FCC).

    Theoretical Value: -1.74756 (per ion pair).
    Reference: https://en.wikipedia.org/wiki/Madelung_constant
    """
    nacl_answer = 1.74756

    L = 2.0
    # FCC primitive vectors
    lattice = (jnp.ones((3, 3)) - jnp.eye(3)) * L / 2

    # Atom (Cation, Na+) at origin
    atom_coords = jnp.array([[0.0, 0.0, 0.0]])
    atom_charges = jnp.array([1.0])

    S = jnp.eye(3).astype(int)

    # Electron (Anion, Cl-) at body center of the conventional cube
    electron_coords = jnp.array([[L / 2, L / 2, L / 2]])

    etot = get_ewald_energy(lattice, atom_coords, atom_charges, S, electron_coords)

    assert abs(etot + nacl_answer) < 1e-4


def test_madelung_nacl_conventional():
    """Test Madelung constant for NaCl using Conventional Cell (Cubic).

    Contains 4 NaCl units. Total energy should be 4 * -1.74756.
    """
    nacl_answer = 1.74756
    L = 2.0

    # Primitive lattice vectors (FCC)
    lattice = (jnp.ones((3, 3)) - jnp.eye(3)) * L / 2
    atom_coords = jnp.array([[0.0, 0.0, 0.0]])
    atom_charges = jnp.array([1.0])

    # Transformation to Conventional Cubic Cell
    # S = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]] yields cubic axes of length L
    S = (jnp.ones((3, 3)) - 2 * jnp.eye(3)).astype(int)

    # Cl positions (Anions/Electrons)
    e1 = jnp.array([L / 2, L / 2, L / 2])
    e2 = jnp.array([L / 2, 0, 0])
    e3 = jnp.array([0, L / 2, 0])
    e4 = jnp.array([0, 0, L / 2])
    electron_coords = jnp.stack([e1, e2, e3, e4])

    etot = get_ewald_energy(lattice, atom_coords, atom_charges, S, electron_coords)

    assert abs(etot / 4 + nacl_answer) < 1e-4


def test_madelung_caf2():
    """Test Madelung constant for CaF2 (Fluorite).

    Theoretical Value: -5.03879 (per CaF2 unit).
    Reference: https://en.wikipedia.org/wiki/Madelung_constant
    """
    caf2_answer = 5.03879
    L = 4 / np.sqrt(3)

    # FCC Lattice
    lattice = (jnp.ones((3, 3)) - jnp.eye(3)) * L / 2

    # Ca2+ at origin
    atom_coords = jnp.array([[0.0, 0.0, 0.0]])
    atom_charges = jnp.array([2.0])  # Ca2+ has q=+2

    S = jnp.eye(3).astype(int)

    # Two F- ions (simulated as electrons)
    e1 = jnp.array([L / 4, L / 4, L / 4])
    e2 = jnp.array([L / 4, -L / 4, L / 4])
    electron_coords = jnp.stack([e1, e2])

    etot = get_ewald_energy(lattice, atom_coords, atom_charges, S, electron_coords)

    assert abs(etot + caf2_answer) < 1e-4
