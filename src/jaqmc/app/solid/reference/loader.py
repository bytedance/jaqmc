# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Loading and validation for solid orbital references."""

import logging
from pathlib import Path

import numpy as np
from upath import UPath

from jaqmc.app.solid.config import SolidConfig

from .model import SolidReference

__all__ = ["load_reference"]

logger = logging.getLogger(__name__)


def load_reference(path: str | Path | UPath, system: SolidConfig) -> SolidReference:
    """Load a solid reference and validate it against the configured system.

    Geometry, primitive lattice, electron-spin counts, and occupied k-point
    boundary phases are enforced as hard errors because orbital evaluation and
    supercell boundary conditions depend on them. Species labels only log a
    warning because external solvers (such as Quantum ESPRESSO) frequently use
    custom labels like ``Fe1`` or ``Fe_up`` for inequivalent magnetic sublattices.

    Returns:
        The loaded reference.

    Raises:
        ValueError: If the geometry, lattice, spin counts, or occupied k-point
            phases do not match the configured solid.
    """
    reference = SolidReference.load(path)
    atoms = system.atoms
    # Hard errors: physical geometry, lattice, spin counts, and Bloch boundary
    # phases must match the QMC simulation cell (with tolerances that allow for
    # text-format rounding in external solver inputs/outputs).
    expected_coords = np.asarray([atom.coords for atom in atoms])
    if expected_coords.shape != reference.atom_coords.shape or not np.allclose(
        expected_coords, reference.atom_coords, atol=1e-4, rtol=0
    ):
        raise ValueError("Reference geometry does not match the configured solid.")
    if not np.allclose(system.lattice_vectors, reference.lattice, atol=1e-5, rtol=0):
        raise ValueError("Reference lattice does not match the configured solid.")
    expected_nspins = tuple(value * system.scale for value in system.electron_spins)
    if expected_nspins != reference.nspins:
        raise ValueError(
            f"Reference nspins={reference.nspins} does not match {expected_nspins}."
        )
    occupied = (reference.alpha_counts > 0) | (reference.beta_counts > 0)
    boundary_phases = reference.kpoints[occupied] @ system.supercell_lattice.T / (
        2 * np.pi
    ) - np.asarray(system.twist)
    if not np.allclose(boundary_phases, np.rint(boundary_phases), rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Reference occupied k-points do not match the configured supercell twist."
        )
    # Soft warning: QE and custom PySCF inputs often rename species (e.g. Fe1/Fe2
    # for broken-symmetry magnetization or Hubbard U) without changing the system.
    expected_symbols = tuple(atom.symbol for atom in atoms)
    if expected_symbols != reference.symbols:
        logger.warning(
            "Reference atoms %s in %s do not match configured solid %s.",
            reference.symbols,
            path,
            expected_symbols,
        )
    return reference
