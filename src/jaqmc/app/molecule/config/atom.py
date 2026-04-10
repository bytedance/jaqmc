# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.atomic import Atom, get_valence_spin_config

from .base import MoleculeConfig

__all__ = ["atom_config"]


def atom_config(
    symbol: str = "H",
    electron_init_width: float = 1.0,
    basis: str = "sto-3g",
    ecp: str | None = None,
):
    """Create a MoleculeConfig for a single atom.

    Args:
        symbol: Element symbol (e.g., "H", "Li", "Fe").
        electron_init_width: Width of Gaussian for electron initialization.
        basis: Basis set name.
        ecp: Effective core potential name. Can be None (no ECP) or
            a string (e.g., "ccecp").

    Returns:
        MoleculeConfig for the specified atom.
    """
    if ecp is not None:
        # With ECP, we need to compute valence electron spins
        # and set the effective charge to match valence electrons
        electron_spins = get_valence_spin_config(symbol, ecp)
        valence_electrons = sum(electron_spins)
        # Set effective charge to number of valence electrons for neutral atom
        atom = Atom(symbol=symbol, coords=[0.0, 0.0, 0.0], charge=valence_electrons)
    else:
        atom = Atom(symbol=symbol, coords=[0.0, 0.0, 0.0])
        electron_spins = atom.spin_config

    return MoleculeConfig(
        atoms=[atom],
        electron_spins=electron_spins,
        electron_init_width=electron_init_width,
        basis=basis,
        ecp=ecp,
    )
