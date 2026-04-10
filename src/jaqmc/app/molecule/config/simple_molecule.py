# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.atomic import Atom

from .base import MoleculeConfig


def benzene():
    atoms = [
        Atom(symbol="C", coords=(0.00000, 2.63664, 0.00000)),
        Atom(symbol="C", coords=(2.28339, 1.31832, 0.00000)),
        Atom(symbol="C", coords=(2.28339, -1.31832, 0.00000)),
        Atom(symbol="C", coords=(0.00000, -2.63664, 0.00000)),
        Atom(symbol="C", coords=(-2.28339, -1.31832, 0.00000)),
        Atom(symbol="C", coords=(-2.28339, 1.31832, 0.00000)),
        Atom(symbol="H", coords=(0.00000, 4.69096, 0.00000)),
        Atom(symbol="H", coords=(4.06250, 2.34549, 0.00000)),
        Atom(symbol="H", coords=(4.06250, -2.34549, 0.00000)),
        Atom(symbol="H", coords=(0.00000, -4.69096, 0.00000)),
        Atom(symbol="H", coords=(-4.06250, -2.34549, 0.00000)),
        Atom(symbol="H", coords=(-4.06250, 2.34549, 0.00000)),
    ]
    # fmt: off
    fixed_spins_per_atom = [
        (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3),
        (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0),
    ]
    # fmt: on
    return MoleculeConfig(
        atoms=atoms,
        electron_spins=_sum_fixed_spins(fixed_spins_per_atom),
        fixed_spins_per_atom=fixed_spins_per_atom,
    )


def ethene():
    atoms = [
        Atom(symbol="C", coords=(0.0, 0.0, 1.26135)),
        Atom(symbol="C", coords=(0.0, 0.0, -1.26135)),
        Atom(symbol="H", coords=(0.0, 1.74390, 2.33889)),
        Atom(symbol="H", coords=(0.0, -1.74390, 2.33889)),
        Atom(symbol="H", coords=(0.0, 1.74390, -2.33889)),
        Atom(symbol="H", coords=(0.0, -1.74390, -2.33889)),
    ]
    fixed_spins_per_atom = [(3, 3), (3, 3), (1, 0), (0, 1), (1, 0), (0, 1)]
    return MoleculeConfig(
        atoms=atoms,
        electron_spins=_sum_fixed_spins(fixed_spins_per_atom),
        fixed_spins_per_atom=fixed_spins_per_atom,
    )


def _sum_fixed_spins(fixed_spins_per_atom: list[tuple[int, int]]) -> tuple[int, int]:
    sums = [sum(components) for components in zip(*fixed_spins_per_atom)]
    return (sums[0], sums[1])
