# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Linear alkane geometry builder."""

import math

from jaqmc.utils.atomic import Atom, AtomInitialization
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR

from .base import AtomConfig, MoleculeConfig

__all__ = ["alkane_config"]

_CC_BOND_LENGTH_ANGSTROM = 1.55
_CH_BOND_LENGTH_ANGSTROM = 1.09
_TETRAHEDRAL_ANGLE_RAD = math.radians(109.4712)


def alkane_config(
    repeat_num: int = 1,
    pp: str | dict[str, str] | None = None,
    electron_init_width: float = 1.0,
) -> MoleculeConfig:
    """Create a linear alkane ``C_(2n) H_(4n+2)`` configuration.

    This reproduces the legacy geometry builder that takes an integer repeat
    count and constructs an all-trans alkane chain. ``repeat_num=1`` gives
    ethane, ``repeat_num=2`` gives butane, and so on.

    Args:
        repeat_num: Number of repeated ``C2H4`` units in the chain.
        pp: Pseudopotential specification. Can be ``None``
            (all-electron), a string (e.g., ``"ccecp"``), or a
            per-element mapping (e.g., ``{"C": "ccecp"}``).
        electron_init_width: Width of Gaussian for electron initialization.

    Returns:
        MoleculeConfig for the requested alkane chain.

    Raises:
        ValueError: If ``repeat_num`` is smaller than 1.
    """
    if repeat_num < 1:
        raise ValueError(f"repeat_num must be at least 1. Got {repeat_num}.")

    dx_c = _CC_BOND_LENGTH_ANGSTROM * math.sin(_TETRAHEDRAL_ANGLE_RAD / 2.0)
    dy_c = _CC_BOND_LENGTH_ANGSTROM * math.cos(_TETRAHEDRAL_ANGLE_RAD / 2.0)
    dx_h = _CH_BOND_LENGTH_ANGSTROM * math.sin(_TETRAHEDRAL_ANGLE_RAD / 2.0)
    dy_h = _CH_BOND_LENGTH_ANGSTROM * math.cos(_TETRAHEDRAL_ANGLE_RAD / 2.0)

    pos_c1 = (-0.5 * dx_c, -0.5 * dy_c, 0.0)
    pos_c2 = (0.5 * dx_c, 0.5 * dy_c, 0.0)
    pos_h11 = (-0.5 * dx_c, -0.5 * dy_c - dy_h, dx_h)
    pos_h12 = (-0.5 * dx_c, -0.5 * dy_c - dy_h, -dx_h)
    pos_h21 = (0.5 * dx_c, 0.5 * dy_c + dy_h, dx_h)
    pos_h22 = (0.5 * dx_c, 0.5 * dy_c + dy_h, -dx_h)
    repeat_displacement = (2 * dx_c, 0.0, 0.0)

    left_terminal_h = (-dx_h - 0.5 * dx_c, dy_h - 0.5 * dy_c, 0.0)
    right_terminal_h = (dx_h + 0.5 * dx_c, -dy_h + 0.5 * dy_c, 0.0)

    carbon_pp = pp.get("C") if isinstance(pp, dict) else pp
    carbon_electrons = Atom("C", [0.0, 0.0, 0.0], pp=carbon_pp).charge
    carbon_spins = (carbon_electrons // 2, carbon_electrons // 2)

    atoms = [_make_atom("H", left_terminal_h, (1, 0))]
    fixed_spins_per_atom = [(1, 0)]

    for i in range(repeat_num):
        shift = _scale_vector(repeat_displacement, i)
        atoms.extend(
            [
                _make_atom("C", _shift(pos_c1, shift), carbon_spins),
                _make_atom("C", _shift(pos_c2, shift), carbon_spins),
                _make_atom("H", _shift(pos_h11, shift), (0, 1)),
                _make_atom("H", _shift(pos_h12, shift), (1, 0)),
                _make_atom("H", _shift(pos_h21, shift), (0, 1)),
                _make_atom("H", _shift(pos_h22, shift), (1, 0)),
            ]
        )
        fixed_spins_per_atom.extend(
            [
                carbon_spins,
                carbon_spins,
                (0, 1),
                (1, 0),
                (0, 1),
                (1, 0),
            ]
        )

    final_shift = _scale_vector(repeat_displacement, repeat_num - 1)
    atoms.append(_make_atom("H", _shift(right_terminal_h, final_shift), (0, 1)))
    fixed_spins_per_atom.append((0, 1))
    spin_imbalance = sum(alpha - beta for alpha, beta in fixed_spins_per_atom)

    return MoleculeConfig(
        atom_configs=atoms,
        s_z=spin_imbalance / 2,
        electron_init_width=electron_init_width,
        pp=pp,
    )


def _make_atom(
    symbol: str,
    coords_angstrom: tuple[float, float, float],
    spin_config: tuple[int, int],
) -> AtomConfig:
    coords_bohr = [coord * ONE_ANGSTROM_IN_BOHR for coord in coords_angstrom]
    return AtomConfig(
        symbol=symbol,
        coords=coords_bohr,
        initialization=AtomInitialization(
            local_s_z=(spin_config[0] - spin_config[1]) / 2,
        ),
    )


def _scale_vector(
    vector: tuple[float, float, float],
    factor: int,
) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _shift(
    coords: tuple[float, float, float],
    displacement: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        coords[0] + displacement[0],
        coords[1] + displacement[1],
        coords[2] + displacement[2],
    )
