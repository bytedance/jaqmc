# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Loading and system validation for solid-state orbital references."""

from pathlib import Path

import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.solid.config import SolidConfig
from jaqmc.utils.supercell import (
    get_primitive_kpts_for_supercell,
    get_reciprocal_vectors,
)

from .model import SolidReference

__all__ = ["load_reference"]


def _same_kpoint_mesh(
    kpoints: np.ndarray, expected_kpoints: np.ndarray, lattice: np.ndarray
) -> bool:
    """Return whether two unordered meshes agree modulo reciprocal vectors."""
    fractional_differences = (
        (np.asarray(kpoints)[:, None, :] - np.asarray(expected_kpoints)[None, :, :])
        @ np.asarray(lattice).T
        / (2 * np.pi)
    )
    matches = np.all(
        np.abs(fractional_differences - np.rint(fractional_differences)) < 1e-7,
        axis=-1,
    )
    return bool(
        np.all(np.sum(matches, axis=1) == 1) and np.all(np.sum(matches, axis=0) == 1)
    )


def load_reference(path: str | Path | UPath, system: SolidConfig) -> SolidReference:
    """Load a solid reference and validate it against its configured system.

    Returns:
        The loaded reference.

    Raises:
        ValueError: If the reference metadata does not match the configured solid.
    """
    reference = SolidReference.load(path)
    atoms = system.atoms
    if tuple(atom.symbol for atom in atoms) != reference.symbols:
        raise ValueError("Reference atoms do not match the configured solid.")
    if not np.allclose(
        np.asarray([atom.coords for atom in atoms]),
        reference.atom_coords,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Reference geometry does not match the configured solid.")
    if not np.allclose(
        system.lattice_vectors,
        reference.lattice,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Reference lattice does not match the configured solid.")
    expected_nspins = tuple(value * system.scale for value in system.electron_spins)
    if expected_nspins != reference.nspins:
        raise ValueError(
            f"Reference nspins={reference.nspins} does not match {expected_nspins}."
        )
    expected_kpoints = np.asarray(
        get_primitive_kpts_for_supercell(
            jnp.asarray(system.supercell_matrix),
            get_reciprocal_vectors(jnp.asarray(system.lattice_vectors)),
            jnp.asarray(system.twist),
        )
    )
    if not _same_kpoint_mesh(reference.kpoints, expected_kpoints, reference.lattice):
        raise ValueError(
            "Reference k-points do not match the configured supercell and twist."
        )
    return reference
