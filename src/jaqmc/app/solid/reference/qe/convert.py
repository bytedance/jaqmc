# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Convert Quantum ESPRESSO save directories to solid references."""

from pathlib import Path

from upath import UPath

from jaqmc.app.solid.reference.model import PlaneWaveSolidReference

from .artifacts import read_qe_calculation
from .reference_builder import build_reference

__all__ = ["convert_save_directory"]


def convert_save_directory(
    input_path: str | Path, output_path: str | Path | UPath
) -> PlaneWaveSolidReference:
    """Convert an unreduced QE 7.x XML/HDF5 ``prefix.save`` directory.

    Only collinear PBE SCF output with ``nosym=.true.`` and ``noinv=.true.`` is
    accepted. Fractional occupations are projected to a single determinant by
    filling the lowest-energy orbitals matching QE's integer electron count.

    Returns:
        The converted plane-wave reference.

    Raises:
        ValueError: If QE output is missing, reduced, unsupported, or inconsistent.
    """
    save_dir = Path(input_path)
    if save_dir.is_file():
        save_dir = save_dir.parent
    if not (save_dir / "data-file-schema.xml").is_file():
        raise ValueError(
            "QE input must be a prefix.save directory containing data-file-schema.xml."
        )
    reference = build_reference(read_qe_calculation(save_dir))
    reference.save(output_path)
    return reference
