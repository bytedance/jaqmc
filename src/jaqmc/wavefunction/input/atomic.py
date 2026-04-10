# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from jaqmc.geometry import obc, pbc
from jaqmc.geometry.pbc import (
    DistanceType,
    SymmetryType,
    get_distance_function,
    get_symmetry_lat,
)


class AtomicEmbedding(TypedDict):
    """Output from MoleculeFeatures or SolidFeatures.

    Attributes:
        ae_features: Flattened atom-electron features for backbone
            (n_elec, n_atoms * (ndim + 1)).
        ee_features: Electron-electron features for backbone
            (n_elec, n_elec, ndim + 1).
        r_ae: Atom-electron distances (n_elec, n_atoms).
        ae_vec: Atom-electron displacement vectors (n_elec, n_atoms, ndim).
    """

    ae_features: Array
    ee_features: Array
    r_ae: Array
    ae_vec: Array


class MoleculeFeatures(nn.Module):
    """Input features for molecular systems (OBC).

    Attributes:
        rescale: If True, make input features grow as log(r) rather than r.
    """

    rescale: bool = False

    def __call__(self, electrons: jnp.ndarray, atoms: jnp.ndarray) -> AtomicEmbedding:
        """Computes features for electrons and atoms.

        Args:
            electrons: Electron positions. Shape (nelectrons, ndim).
            atoms: Atom positions. Shape (natoms, ndim).

        Returns:
            Embedding containing backbone features and envelope distances.
        """
        ee_vec, r_ee = obc.pair_displacements_within(electrons)
        ae_vec, r_ae = obc.pair_displacements_between(electrons, atoms)

        if self.rescale:
            log_r_ae = jnp.log(1 + r_ae)[..., None]
            ae_features = jnp.concatenate(
                (log_r_ae, ae_vec * log_r_ae / r_ae[..., None]), axis=2
            )
            log_r_ee = jnp.log(1 + r_ee)[..., None]
            ee_features = jnp.concatenate(
                (log_r_ee, ee_vec * log_r_ee / r_ee[..., None]), axis=2
            )
        else:
            ae_features = jnp.concatenate((r_ae[..., None], ae_vec), axis=2)
            ee_features = jnp.concatenate((r_ee[..., None], ee_vec), axis=2)

        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])

        return AtomicEmbedding(
            ae_features=ae_features,
            ee_features=ee_features,
            r_ae=r_ae,
            ae_vec=ae_vec,
        )


class SolidFeatures(nn.Module):
    """Input features for periodic systems (solids).

    Attributes:
        simulation_lattice: Lattice vectors of the simulation cell (nelectrons).
        primitive_lattice: Lattice vectors of the primitive cell (natoms).
        distance_type: Type of periodic distance to use ('nu' or 'tri').
        sym_type: Symmetry type for auxiliary lattice vectors.
    """

    simulation_lattice: jnp.ndarray
    primitive_lattice: jnp.ndarray
    distance_type: DistanceType = DistanceType.nu
    sym_type: SymmetryType = SymmetryType.minimal

    def setup(self):
        """Precompute symmetry-reduced lattice vectors for distance evaluation."""
        self.sim_av, self.sim_bv = get_symmetry_lat(
            self.simulation_lattice, self.sym_type
        )
        self.prim_av, self.prim_bv = get_symmetry_lat(
            self.primitive_lattice, self.sym_type
        )

    def __call__(self, electrons: jnp.ndarray, atoms: jnp.ndarray) -> AtomicEmbedding:
        """Computes periodic features for electrons and atoms.

        Args:
            electrons: Electron positions. Shape (nelectrons, ndim).
            atoms: Atom positions. Shape (natoms, ndim).

        Returns:
            Embedding containing backbone features and envelope distances.
        """
        distance_func = get_distance_function(self.distance_type)

        # Wrap electrons to primitive cell for e-n features
        prim_electrons = pbc.wrap_positions(electrons, self.primitive_lattice)
        ae_displacements = prim_electrons[:, None, :] - atoms
        r_ae, ae_vec = distance_func(ae_displacements, self.prim_av, self.prim_bv)

        # Wrap electrons to simulation cell for e-e features
        sim_electrons = pbc.wrap_positions(electrons, self.simulation_lattice)
        ee_displacements = sim_electrons[:, None, :] - sim_electrons[None, :, :]

        n = electrons.shape[0]
        eye = jnp.eye(n)[..., None]
        r_ee, ee_vec = distance_func(ee_displacements + eye, self.sim_av, self.sim_bv)

        # Mask out diagonal for e-e distances
        r_ee = r_ee * (1.0 - jnp.eye(n))
        ee_vec = ee_vec * (1.0 - jnp.eye(n))[..., None]

        # Prepare features in jaqmc format (r, vec)
        ae_features = jnp.concatenate([r_ae[..., None], ae_vec], axis=-1)
        ae_features = jnp.reshape(ae_features, [n, -1])
        ee_features = jnp.concatenate([r_ee[..., None], ee_vec], axis=-1)

        return AtomicEmbedding(
            ae_features=ae_features,
            ee_features=ee_features,
            r_ae=r_ae,
            ae_vec=ae_vec,
        )
