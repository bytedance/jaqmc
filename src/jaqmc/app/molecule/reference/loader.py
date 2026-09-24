# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Loading and system validation for molecular orbital references."""

from pathlib import Path

import numpy as np
from upath import UPath

from jaqmc.app.molecule.config import MoleculeConfig

from .model import MoleculeReference

__all__ = ["load_reference"]


def load_reference(
    path: str | Path | UPath, system: MoleculeConfig
) -> MoleculeReference:
    """Load a molecular reference and validate it against the configured system.

    Returns:
        The loaded reference.

    Raises:
        ValueError: If the reference metadata does not match the configured molecule.
    """
    reference = MoleculeReference.load(path)
    atoms = system.atoms
    if tuple(atom.symbol for atom in atoms) != reference.symbols:
        raise ValueError("Reference atoms do not match the configured molecule.")
    if not np.allclose(
        np.asarray([atom.coords for atom in atoms]),
        reference.coords,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Reference geometry does not match the configured molecule.")
    if not np.array_equal([atom.charge for atom in atoms], reference.charges):
        raise ValueError(
            "Reference effective charges do not match the configured molecule."
        )
    if tuple(system.electron_spins) != reference.nspins:
        raise ValueError(
            "Reference electron spins do not match the configured molecule."
        )
    return reference
