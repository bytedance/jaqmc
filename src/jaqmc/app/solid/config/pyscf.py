# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PySCF solver configuration for solid orbital references."""

from dataclasses import field
from typing import Any, Literal

import serde

from jaqmc.utils.config import configurable_dataclass

__all__ = ["SolidPySCFSolverConfig"]


@configurable_dataclass
class SolidPySCFSolverConfig:
    """Configuration for a generated periodic PySCF reference job.

    Args:
        basis: PySCF basis specification passed to ``pyscf.pbc.gto.Cell``.
            A string names one basis for every element. ``None`` or an empty
            mapping uses the automatic double-zeta policy (``cc-pVDZ`` for
            all-electron elements and ``ccecpccpvdz`` for ``ccecp``). A mapping
            overrides selected elements and fills the rest from that policy.
            Other ECP families need an explicit entry.
        pp: PySCF pseudopotential selection for the generated job. A string
            applies one pseudopotential to every element; a mapping overrides
            selected elements, while omitted elements inherit ``system.pp``.
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
        extra: Extra solver keys flattened onto ``solver.*`` and forwarded to
            the selected PySCF mean-field object.
    """

    basis: str | dict[str, str] | None = None
    pp: str | dict[str, str] = field(default_factory=dict)
    method: Literal["KRHF", "KUHF", "KRKS", "KUKS"] = "KUHF"
    xc: str = "pbe"
    checkpoint: str = "pyscf.chk"
    verbose: int = 4
    extra: dict[str, Any] = serde.field(flatten=True, default_factory=dict)
