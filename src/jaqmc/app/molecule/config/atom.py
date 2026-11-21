# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.atomic import elements
from jaqmc.utils.config import configurable_dataclass

from .base import AtomConfig, MoleculeConfig

__all__ = ["AtomBuilder"]


def _validate_symbol(symbol: str) -> None:
    if symbol not in elements.from_symbol:
        raise ValueError(f"Unknown element symbol {symbol!r}.")


@configurable_dataclass
class AtomBuilder:
    """Builder for a one-atom molecular system.

    Args:
        symbol: Element symbol (e.g., "H", "Li", "Fe").
        electron_init_width: Width of Gaussian for electron initialization.
        pp: Pseudopotential for this atom. ``None`` means all-electron;
            otherwise the pseudopotential name (e.g., ``"ccecp"``, ``"ph"``).

    """

    symbol: str = "H"
    electron_init_width: float = 1.0
    pp: str | None = None

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)

    def __call__(self) -> MoleculeConfig:
        _validate_symbol(self.symbol)
        atom = AtomConfig(symbol=self.symbol, coords=[0.0, 0.0, 0.0])
        return MoleculeConfig(
            atoms=[atom],
            s_z=elements.from_symbol[self.symbol].unpaired_electron / 2,
            electron_init_width=self.electron_init_width,
            pp={self.symbol: self.pp} if self.pp is not None else None,
        )
