# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Utilities for querying effective core potential (ECP) electron counts."""

from collections.abc import Sequence

from . import elements
from .atom import Atom

__all__ = ["get_core_electrons", "get_valence_spin_config"]


def get_valence_spin_config(symbol: str, ecp: str | dict[str, str]) -> tuple[int, int]:
    """Get valence electron spin configuration for an element with an ECP.

    Builds a temporary PySCF molecule to determine the number of valence
    electrons after core electrons are replaced by the ECP. Returns
    ``mol.nelec`` directly, which splits the valence electrons into
    ``(n_alpha, n_beta)`` based on the element's ground-state spin.

    Args:
        symbol: Element symbol (e.g., ``"Fe"``).
        ecp: ECP name (e.g., ``"ccecp"``) or per-element mapping.

    Returns:
        Tuple of (n_alpha, n_beta) valence electrons.
    """
    import pyscf.gto

    mol = pyscf.gto.Mole(atom=[[symbol, [0.0, 0.0, 0.0]]], unit="bohr")
    mol.ecp = ecp
    mol.spin = elements.from_symbol[symbol].unpaired_electron
    mol.build()

    return mol.nelec


def get_core_electrons(
    atoms: Sequence[Atom],
    ecp: str | dict[str, str] | None,
) -> dict[str, int]:
    """Get the number of core electrons removed by ECP for each element.

    Args:
        atoms: List of Atom objects.
        ecp: ECP specification (name, per-element mapping, or ``None``).

    Returns:
        Mapping from element symbol to number of core electrons removed.
        Elements without core electrons removed are omitted.
    """
    if ecp is None:
        return {}

    core: dict[str, int] = {}
    for symbol in {atom.symbol for atom in atoms}:
        total = elements.from_symbol[symbol].atomic_number
        valence = sum(get_valence_spin_config(symbol, ecp))
        if valence < total:
            core[symbol] = total - valence

    return core
