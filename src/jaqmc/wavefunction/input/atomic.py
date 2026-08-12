# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypedDict

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from jaqmc.geometry import obc, pbc
from jaqmc.geometry.pbc import (
    DistanceType,
    SymmetryType,
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
    """Input features for periodic systems (solids and moire).

    The two-electron stream always uses simulation-cell electron-electron
    features. The one-electron stream is periodic electron-atom features when
    ``atoms`` is supplied (solids), or periodic electron-origin features when it
    is not (moire systems, which have no explicit atoms).

    Attributes:
        simulation_lattice: Lattice vectors of the simulation cell (nelectrons).
        primitive_lattice: Lattice vectors of the primitive cell (natoms).
        ae_lattice: Lattice used by the one-electron stream. ``"primitive"``
            (default) matches the solid convention; ``"simulation"`` is used by
            moire fractional filling states.
        distance_type: Periodic distance representation. ``tri`` (default) uses
            trigonometric features with 6D displacement vectors; ``nu`` uses
            polynomial features with 3D displacement vectors.
        sym_type: Symmetry type for auxiliary lattice vectors.
    """

    simulation_lattice: jnp.ndarray
    primitive_lattice: jnp.ndarray
    ae_lattice: Literal["primitive", "simulation"] = "primitive"
    distance_type: DistanceType = DistanceType.tri
    sym_type: SymmetryType = SymmetryType.minimal

    def setup(self):
        """Precompute symmetry-reduced lattice vectors for distance evaluation.

        Raises:
            ValueError: If ``ae_lattice`` is not ``"primitive"`` or
                ``"simulation"``.
        """
        self.sim_av, self.sim_bv = get_symmetry_lat(
            self.simulation_lattice, self.sym_type
        )
        self.prim_av, self.prim_bv = get_symmetry_lat(
            self.primitive_lattice, self.sym_type
        )
        if self.ae_lattice == "simulation":
            self.ae_av, self.ae_bv = self.sim_av, self.sim_bv
            self.ae_lat = self.simulation_lattice
        elif self.ae_lattice == "primitive":
            self.ae_av, self.ae_bv = self.prim_av, self.prim_bv
            self.ae_lat = self.primitive_lattice
        else:
            raise ValueError(
                "ae_lattice must be either 'primitive' or 'simulation', "
                f"got {self.ae_lattice!r}"
            )

    def __call__(
        self, electrons: jnp.ndarray, atoms: jnp.ndarray | None = None
    ) -> AtomicEmbedding:
        """Computes periodic features for electrons and (optionally) atoms.

        Args:
            electrons: Electron positions. Shape (nelectrons, ndim).
            atoms: Atom positions. Shape (natoms, ndim). When ``None`` the
                one-electron stream uses electron-origin features, as required
                by moire systems.

        Returns:
            Embedding containing backbone features and envelope distances.
        """
        # One-electron stream: electron-atom, or electron-origin when atomless.
        distance_fn = pbc.get_distance_function(self.distance_type)
        prim_electrons = pbc.wrap_positions(electrons, self.ae_lat)
        if atoms is None:
            ae_displacements = prim_electrons[:, None, :]
        else:
            ae_displacements = prim_electrons[:, None, :] - atoms
        r_ae, ae_vec = distance_fn(ae_displacements, self.ae_av, self.ae_bv)

        # Wrap electrons to simulation cell for e-e features. Add the identity
        # before evaluating the distance to avoid a zero-distance singularity on
        # the diagonal (self pairs), then zero the diagonal of the outputs.
        n = electrons.shape[0]
        sim_electrons = pbc.wrap_positions(electrons, self.simulation_lattice)
        ee_displacements = sim_electrons[:, None, :] - sim_electrons[None, :, :]
        r_ee, ee_vec = distance_fn(
            ee_displacements + jnp.eye(n)[..., None], self.sim_av, self.sim_bv
        )
        mask = 1.0 - jnp.eye(n)
        r_ee = r_ee * mask
        ee_vec = ee_vec * mask[..., None]

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
