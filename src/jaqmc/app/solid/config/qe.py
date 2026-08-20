# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Quantum ESPRESSO solver configuration for solid orbital references."""

from typing import Literal

import serde

from jaqmc.utils.config import configurable_dataclass

__all__ = ["SolidQESolverConfig"]


@configurable_dataclass
class SolidQESolverConfig:
    """Configuration for a generated Quantum ESPRESSO reference job.

    Args:
        pw_cutoff: Plane-wave kinetic-energy cutoff in Rydberg
            (``ecutwfc``).
        rho_cutoff: Charge-density cutoff in Rydberg (``ecutrho``).
        pseudo_dir: Directory containing compatible UPF files. QE will not
            download pseudopotentials.
        prefix: Quantum ESPRESSO prefix. Conversion reads ``{prefix}.save``.
        smearing: ``"off"`` writes fixed occupations. Other values select the
            corresponding QE smearing function.
        degauss: Smearing width in Rydberg when ``smearing`` is not ``"off"``;
            must be positive when smearing is enabled.
        pseudo_file: Per-element UPF filenames. Required for every element;
            JaQMC does not infer filenames from ``system.pp``.
    """

    pw_cutoff: float = 80.0
    rho_cutoff: float = 320.0
    pseudo_dir: str | None = None
    prefix: str = "jaqmc"
    smearing: Literal["off", "gauss", "mp", "mv", "fd"] = "off"
    degauss: float = 0.01
    pseudo_file: dict[str, str] = serde.field(default_factory=dict)
