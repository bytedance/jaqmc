# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Literal

from jaqmc.utils.atomic import Atom
from jaqmc.utils.atomic.atomic_system import AtomicSystemConfig
from jaqmc.utils.atomic.pretrain import PretrainReferenceConfig
from jaqmc.utils.config import configurable_dataclass

__all__ = ["MoleculeConfig", "MoleculePretrainReferenceConfig"]


@configurable_dataclass
class MoleculeConfig(AtomicSystemConfig):
    atoms: list[Atom] = field(default_factory=lambda: [Atom("H", [0, 0, 0])])
    electron_spins: tuple[int, int] = (1, 0)
    fixed_spins_per_atom: list[tuple[int, int]] | None = None
    electron_init_width: float = 1.0


@configurable_dataclass
class MoleculePretrainReferenceConfig(PretrainReferenceConfig):
    method: Literal["RHF", "UHF"] = "UHF"
    "Variants of Hartree-Fock method."
