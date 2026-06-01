# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.atomic import make_atom

from .base import MoleculeConfig

__all__ = ["atom_config"]


def atom_config(
    symbol: str = "H",
    electron_init_width: float = 1.0,
    pp: str | None = None,
):
    """Create a MoleculeConfig for a single atom.

    Args:
        symbol: Element symbol (e.g., "H", "Li", "Fe").
        electron_init_width: Width of Gaussian for electron initialization.
        pp: Pseudopotential for this atom. ``None`` means all-electron;
            otherwise the pseudopotential name (e.g., ``"ccecp"``, ``"ph"``).

    Returns:
        MoleculeConfig for the specified atom.
    """
    atom = make_atom(symbol, [0.0, 0.0, 0.0], pp=pp)
    return MoleculeConfig(
        atoms=[atom],
        electron_spins=atom.spin_config,
        electron_init_width=electron_init_width,
        pp={symbol: pp} if pp is not None else None,
    )
