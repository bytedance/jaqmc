# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PySCF solver configuration for solid orbital references."""

from dataclasses import field
from typing import Literal

from jaqmc.utils.config import configurable_dataclass

__all__ = ["SolidPySCFSolverConfig"]


@configurable_dataclass
class SolidPySCFSolverConfig:
    """Configuration for a generated periodic PySCF reference job.

    Args:
        basis: PySCF basis specification passed to ``pyscf.pbc.gto.Cell``.
            A string names one basis for every element. ``None`` or an empty
            mapping uses the automatic double-zeta policy (``cc-pVDZ`` for
            all-electron elements and ``ccecpccpvdz`` for ``ccecp`` and PH).
            A mapping overrides selected elements and fills the rest from that
            policy. Other ECP families need an explicit entry.
        pp: PySCF pseudopotential selection for the generated job. A string
            applies one pseudopotential to every element; a mapping overrides
            selected elements, while omitted elements inherit ``system.pp``
            (with PH translated to its ``ccecp`` surrogate).
            These choices do not modify ``system.pp``. Each override may name
            a PySCF ECP or GTH pseudopotential and must preserve the
            valence-electron count from ``system.pp``.
        method: Periodic PySCF mean-field method. ``KRHF`` and ``KUHF`` use
            :mod:`pyscf.pbc.scf`; ``KRKS`` and ``KUKS`` use
            :mod:`pyscf.pbc.dft`.
        xc: Exchange-correlation functional assigned for DFT methods.
        checkpoint: Checkpoint filename written inside the generated job
            directory.
        verbose: PySCF verbosity level.
        conv_tol: SCF energy convergence threshold in Hartree.
        conv_tol_grad: SCF orbital-gradient convergence threshold. ``None`` uses
            PySCF's default (``sqrt(conv_tol)``).
        max_cycle: Maximum number of SCF iterations.
        diis_space: Number of Fock matrices retained in the DIIS subspace.
        level_shift: Level shift (in Hartree) applied to virtual orbitals.
        damp: Fock-matrix damping factor between 0 and 1.
        init_guess: Initial SCF density-matrix guess method (for example
            ``"minao"``, ``"atom"``, or ``"1e"``).
    """

    basis: str | dict[str, str] | None = None
    pp: str | dict[str, str] = field(default_factory=dict)
    method: Literal["KRHF", "KUHF", "KRKS", "KUKS"] = "KUHF"
    xc: str = "pbe"
    checkpoint: str = "pyscf.chk"
    verbose: int = 4
    conv_tol: float = 1e-7
    conv_tol_grad: float | None = None
    max_cycle: int = 50
    diis_space: int = 8
    level_shift: float = 0.0
    damp: float = 0.0
    init_guess: str = "minao"
