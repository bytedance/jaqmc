# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Loading and validation for molecular orbital references."""

import logging
from pathlib import Path

import numpy as np
from upath import UPath

from jaqmc.app.molecule.config import MoleculeConfig

from .model import MoleculeReference

__all__ = ["load_reference"]

logger = logging.getLogger(__name__)


def load_reference(
    path: str | Path | UPath, system: MoleculeConfig
) -> MoleculeReference:
    """Load a molecular reference and validate it against the configured system.

    Geometry and electron-spin counts are enforced as hard errors because AO
    basis functions are centered at ``reference.coords`` and orbital shapes
    depend on ``nspins``. Species labels and effective charges only log
    warnings so custom reference setups (such as solver-specific atom labels or
    pseudopotential tweaks) are not blocked.

    Returns:
        The loaded reference.

    Raises:
        ValueError: If the reference geometry or electron-spin counts do not
            match the configured molecule.
    """
    reference = MoleculeReference.load(path)
    atoms = system.atoms
    # Hard errors: mismatched nuclear positions or spin counts make orbital
    # evaluation incompatible with the QMC geometry/wavefunction, while 1e-4
    # Bohr tolerates text-format rounding in external solver inputs.
    expected_coords = np.asarray([atom.coords for atom in atoms])
    if expected_coords.shape != reference.coords.shape or not np.allclose(
        expected_coords, reference.coords, atol=1e-4, rtol=0
    ):
        raise ValueError("Reference geometry does not match the configured molecule.")
    if tuple(system.electron_spins) != reference.nspins:
        raise ValueError(
            "Reference electron spins do not match the configured molecule."
        )
    # Soft warnings: species labels and effective charges can legitimately
    # differ in custom pretraining setups without invalidating the orbitals.
    expected_symbols = tuple(atom.symbol for atom in atoms)
    if expected_symbols != reference.symbols:
        logger.warning(
            "Reference atoms %s in %s do not match configured molecule %s.",
            reference.symbols,
            path,
            expected_symbols,
        )
    expected_charges = [atom.charge for atom in atoms]
    if not np.array_equal(expected_charges, reference.charges):
        logger.warning(
            "Reference effective charges %s in %s do not match configured molecule %s.",
            reference.charges.tolist(),
            path,
            expected_charges,
        )
    return reference
