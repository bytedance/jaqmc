# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

import collections
from dataclasses import dataclass, field


@dataclass
class Element:
    """Chemical element.

    Attributes:
      symbol: official symbol of element.
      atomic_number: atomic number of element.
      period: period to which the element belongs.
      spin: overrides default ground-state spin-configuration based on the
        element's group (main groups only).
    """

    symbol: str
    atomic_number: int
    period: int
    group: int
    spin: int | None = field(default=None, repr=False)

    @property
    def unpaired_electron(self) -> int:
        """Canonical spin configuration (via Hund's rules) of neutral atom.

        Returns:
            Number of unpaired electrons (as required by PySCF) in the neutral
            atom's ground state.

        Raises:
            NotImplementedError: If the element is a transition metal and the
                spin value is not set at initialization.

        Examples:
            >>> from jaqmc.utils.atomic.elements import SYMBOLS
            >>> SYMBOLS['H'].unpaired_electron
            1
            >>> SYMBOLS['C'].unpaired_electron
            2
            >>> SYMBOLS['Ne'].unpaired_electron
            0
        """
        if self.spin is not None:
            return self.spin
        unpaired = {1: 1, 2: 0, 13: 1, 14: 2, 15: 3, 16: 2, 17: 1, 18: 0}
        if self.group in unpaired:
            return unpaired[self.group]
        else:
            raise NotImplementedError(
                "Spin configuration for transition metals not set."
            )


# Atomic symbols for all known elements
_ELEMENTS = (
    Element(symbol="X", atomic_number=0, period=0, group=-86),
    Element(symbol="H", atomic_number=1, period=1, group=1),
    Element(symbol="He", atomic_number=2, period=1, group=18),
    Element(symbol="Li", atomic_number=3, period=2, group=1),
    Element(symbol="Be", atomic_number=4, period=2, group=2),
    Element(symbol="B", atomic_number=5, period=2, group=13),
    Element(symbol="C", atomic_number=6, period=2, group=14),
    Element(symbol="N", atomic_number=7, period=2, group=15),
    Element(symbol="O", atomic_number=8, period=2, group=16),
    Element(symbol="F", atomic_number=9, period=2, group=17),
    Element(symbol="Ne", atomic_number=10, period=2, group=18),
    Element(symbol="Na", atomic_number=11, period=3, group=1),
    Element(symbol="Mg", atomic_number=12, period=3, group=2),
    Element(symbol="Al", atomic_number=13, period=3, group=13),
    Element(symbol="Si", atomic_number=14, period=3, group=14),
    Element(symbol="P", atomic_number=15, period=3, group=15),
    Element(symbol="S", atomic_number=16, period=3, group=16),
    Element(symbol="Cl", atomic_number=17, period=3, group=17),
    Element(symbol="Ar", atomic_number=18, period=3, group=18),
    Element(symbol="K", atomic_number=19, period=4, group=1),
    Element(symbol="Ca", atomic_number=20, period=4, group=2),
    Element(symbol="Sc", atomic_number=21, period=4, group=3, spin=1),
    Element(symbol="Ti", atomic_number=22, period=4, group=4, spin=2),
    Element(symbol="V", atomic_number=23, period=4, group=5, spin=3),
    Element(symbol="Cr", atomic_number=24, period=4, group=6, spin=6),
    Element(symbol="Mn", atomic_number=25, period=4, group=7, spin=5),
    Element(symbol="Fe", atomic_number=26, period=4, group=8, spin=4),
    Element(symbol="Co", atomic_number=27, period=4, group=9, spin=3),
    Element(symbol="Ni", atomic_number=28, period=4, group=10, spin=2),
    Element(symbol="Cu", atomic_number=29, period=4, group=11, spin=1),
    Element(symbol="Zn", atomic_number=30, period=4, group=12, spin=0),
    Element(symbol="Ga", atomic_number=31, period=4, group=13),
    Element(symbol="Ge", atomic_number=32, period=4, group=14),
    Element(symbol="As", atomic_number=33, period=4, group=15),
    Element(symbol="Se", atomic_number=34, period=4, group=16),
    Element(symbol="Br", atomic_number=35, period=4, group=17),
    Element(symbol="Kr", atomic_number=36, period=4, group=18),
    Element(symbol="Rb", atomic_number=37, period=5, group=1),
    Element(symbol="Sr", atomic_number=38, period=5, group=2),
    Element(symbol="Y", atomic_number=39, period=5, group=3, spin=1),
    Element(symbol="Zr", atomic_number=40, period=5, group=4, spin=2),
    Element(symbol="Nb", atomic_number=41, period=5, group=5, spin=5),
    Element(symbol="Mo", atomic_number=42, period=5, group=6, spin=6),
    Element(symbol="Tc", atomic_number=43, period=5, group=7, spin=5),
    Element(symbol="Ru", atomic_number=44, period=5, group=8, spin=4),
    Element(symbol="Rh", atomic_number=45, period=5, group=9, spin=3),
    Element(symbol="Pd", atomic_number=46, period=5, group=10, spin=0),
    Element(symbol="Ag", atomic_number=47, period=5, group=11, spin=1),
    Element(symbol="Cd", atomic_number=48, period=5, group=12, spin=0),
    Element(symbol="In", atomic_number=49, period=5, group=13),
    Element(symbol="Sn", atomic_number=50, period=5, group=14),
    Element(symbol="Sb", atomic_number=51, period=5, group=15),
    Element(symbol="Te", atomic_number=52, period=5, group=16),
    Element(symbol="I", atomic_number=53, period=5, group=17),
    Element(symbol="Xe", atomic_number=54, period=5, group=18),
    Element(symbol="Cs", atomic_number=55, period=6, group=1),
    Element(symbol="Ba", atomic_number=56, period=6, group=2),
    Element(symbol="La", atomic_number=57, period=6, group=3),
    Element(symbol="Ce", atomic_number=58, period=6, group=-1),
    Element(symbol="Pr", atomic_number=59, period=6, group=-1),
    Element(symbol="Nd", atomic_number=60, period=6, group=-1),
    Element(symbol="Pm", atomic_number=61, period=6, group=-1),
    Element(symbol="Sm", atomic_number=62, period=6, group=-1),
    Element(symbol="Eu", atomic_number=63, period=6, group=-1),
    Element(symbol="Gd", atomic_number=64, period=6, group=-1),
    Element(symbol="Tb", atomic_number=65, period=6, group=-1),
    Element(symbol="Dy", atomic_number=66, period=6, group=-1),
    Element(symbol="Ho", atomic_number=67, period=6, group=-1),
    Element(symbol="Er", atomic_number=68, period=6, group=-1),
    Element(symbol="Tm", atomic_number=69, period=6, group=-1),
    Element(symbol="Yb", atomic_number=70, period=6, group=-1),
    Element(symbol="Lu", atomic_number=71, period=6, group=-1),
    Element(symbol="Hf", atomic_number=72, period=6, group=4),
    Element(symbol="Ta", atomic_number=73, period=6, group=5),
    Element(symbol="W", atomic_number=74, period=6, group=6),
    Element(symbol="Re", atomic_number=75, period=6, group=7),
    Element(symbol="Os", atomic_number=76, period=6, group=8),
    Element(symbol="Ir", atomic_number=77, period=6, group=9),
    Element(symbol="Pt", atomic_number=78, period=6, group=10),
    Element(symbol="Au", atomic_number=79, period=6, group=11),
    Element(symbol="Hg", atomic_number=80, period=6, group=12),
    Element(symbol="Tl", atomic_number=81, period=6, group=13),
    Element(symbol="Pb", atomic_number=82, period=6, group=14),
    Element(symbol="Bi", atomic_number=83, period=6, group=15),
    Element(symbol="Po", atomic_number=84, period=6, group=16),
    Element(symbol="At", atomic_number=85, period=6, group=17),
    Element(symbol="Rn", atomic_number=86, period=6, group=18),
    Element(symbol="Fr", atomic_number=87, period=7, group=1),
    Element(symbol="Ra", atomic_number=88, period=7, group=2),
    Element(symbol="Ac", atomic_number=89, period=7, group=3),
    Element(symbol="Th", atomic_number=90, period=7, group=-1),
    Element(symbol="Pa", atomic_number=91, period=7, group=-1),
    Element(symbol="U", atomic_number=92, period=7, group=-1),
    Element(symbol="Np", atomic_number=93, period=7, group=-1),
    Element(symbol="Pu", atomic_number=94, period=7, group=-1),
    Element(symbol="Am", atomic_number=95, period=7, group=-1),
    Element(symbol="Cm", atomic_number=96, period=7, group=-1),
    Element(symbol="Bk", atomic_number=97, period=7, group=-1),
    Element(symbol="Cf", atomic_number=98, period=7, group=-1),
    Element(symbol="Es", atomic_number=99, period=7, group=-1),
    Element(symbol="Fm", atomic_number=100, period=7, group=-1),
    Element(symbol="Md", atomic_number=101, period=7, group=-1),
    Element(symbol="No", atomic_number=102, period=7, group=-1),
    Element(symbol="Lr", atomic_number=103, period=7, group=-1),
    Element(symbol="Rf", atomic_number=104, period=7, group=4),
    Element(symbol="Db", atomic_number=105, period=7, group=5),
    Element(symbol="Sg", atomic_number=106, period=7, group=6),
    Element(symbol="Bh", atomic_number=107, period=7, group=7),
    Element(symbol="Hs", atomic_number=108, period=7, group=8),
    Element(symbol="Mt", atomic_number=109, period=7, group=9),
    Element(symbol="Ds", atomic_number=110, period=7, group=10),
    Element(symbol="Rg", atomic_number=111, period=7, group=11),
    Element(symbol="Cn", atomic_number=112, period=7, group=12),
    Element(symbol="Nh", atomic_number=113, period=7, group=13),
    Element(symbol="Fl", atomic_number=114, period=7, group=14),
    Element(symbol="Mc", atomic_number=115, period=7, group=15),
    Element(symbol="Lv", atomic_number=116, period=7, group=16),
    Element(symbol="Ts", atomic_number=117, period=7, group=17),
    Element(symbol="Og", atomic_number=118, period=7, group=18),
)


# Lookup by atomic number.
from_atomic_number = {element.atomic_number: element for element in _ELEMENTS}
# Lookup by symbol.
from_symbol = {element.symbol: element for element in _ELEMENTS}

ATOMIC_NUMS = {element.atomic_number: element for element in _ELEMENTS}
# Lookup by symbol instead of atomic number.
SYMBOLS = {element.symbol: element for element in _ELEMENTS}
# Lookup by period.
PERIODS = collections.defaultdict(list)
for element in _ELEMENTS:
    PERIODS[element.period].append(element)
