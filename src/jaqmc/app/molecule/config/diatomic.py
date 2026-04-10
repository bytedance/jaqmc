# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import re

from jaqmc.utils.atomic import Atom, elements, get_valence_spin_config
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit

from .base import MoleculeConfig

__all__ = ["diatomic_config"]


def _parse_diatomic_formula(formula: str) -> tuple[str, str]:
    """Parse a diatomic chemical formula into two element symbols.

    Supports homonuclear (e.g., ``"H2"``, ``"N2"``, ``"Li2"``) and
    heteronuclear (e.g., ``"LiH"``, ``"CN"``, ``"ClF"``) formulas.

    Args:
        formula: Chemical formula string.

    Returns:
        Tuple of two element symbols.

    Raises:
        ValueError: If the formula cannot be parsed as a diatomic molecule
            or contains unknown element symbols.
    """
    # Homonuclear: formula ends with "2" and prefix is a valid element
    if formula.endswith("2"):
        symbol = formula[:-1]
        if symbol in elements.from_symbol:
            return (symbol, symbol)

    # Heteronuclear: split into element symbols
    symbols = re.findall(r"[A-Z][a-z]*", formula)
    if len(symbols) != 2 or "".join(symbols) != formula:
        raise ValueError(
            f"Cannot parse {formula!r} as a diatomic formula. "
            f"Expected a formula like 'H2', 'LiH', or 'ClF'."
        )
    for s in symbols:
        if s not in elements.from_symbol:
            raise ValueError(f"Unknown element symbol {s!r} in formula {formula!r}.")

    return (symbols[0], symbols[1])


def diatomic_config(
    formula: str = "H2",
    bond_length: float = 1.4,
    unit: LengthUnit = LengthUnit.bohr,
    basis: str | dict[str, str] = "sto-3g",
    ecp: str | dict[str, str] | None = None,
    spin: int = 0,
    electron_init_width: float = 1.0,
):
    """Create a MoleculeConfig for a diatomic molecule.

    The two atoms are placed along the z-axis, centered at the origin.

    Args:
        formula: Chemical formula (e.g., ``"H2"``, ``"LiH"``, ``"N2"``,
            ``"ClF"``). Homonuclear diatomics use the ``"X2"`` convention.
        bond_length: Distance between the two atoms.
        unit: Length unit for ``bond_length`` and atom coordinates.
            Either ``"bohr"`` or ``"angstrom"``.
        basis: Basis set name, or per-element mapping
            (e.g., ``{"Li": "ccecpccpvdz", "H": "cc-pvdz"}``).
        ecp: Effective core potential specification. Can be ``None``
            (all-electron), a string (e.g., ``"ccecp"``), or a
            per-element mapping (e.g., ``{"Li": "ccecp"}``).
        spin: Total spin (number of unpaired electrons). Defaults to 0
            (singlet).
        electron_init_width: Width of Gaussian for electron initialization.

    Returns:
        MoleculeConfig for the diatomic molecule.

    Raises:
        ValueError: If ``formula`` cannot be parsed as a diatomic, or
            if total electrons and ``spin`` have different parity.
    """
    sym1, sym2 = _parse_diatomic_formula(formula)

    if unit == LengthUnit.angstrom:
        bond_length *= ONE_ANGSTROM_IN_BOHR
    half = bond_length / 2
    atoms = []
    total_electrons = 0

    for i, symbol in enumerate((sym1, sym2)):
        sign = -1 if i == 0 else 1
        coord = [0.0, 0.0, sign * half]

        if ecp is not None:
            valence = sum(get_valence_spin_config(symbol, ecp))
            atom = Atom(symbol=symbol, coords=coord, charge=valence)
            total_electrons += valence
        else:
            atom = Atom(symbol=symbol, coords=coord)
            total_electrons += elements.from_symbol[symbol].atomic_number

        atoms.append(atom)

    if (total_electrons + spin) % 2 != 0:
        raise ValueError(
            f"Total electrons ({total_electrons}) and spin ({spin}) "
            f"must have the same parity."
        )

    n_alpha = (total_electrons + spin) // 2
    n_beta = (total_electrons - spin) // 2

    return MoleculeConfig(
        atoms=atoms,
        electron_spins=(n_alpha, n_beta),
        electron_init_width=electron_init_width,
        basis=basis,
        ecp=ecp,
    )
