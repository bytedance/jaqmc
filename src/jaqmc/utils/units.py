# Copyright 2019 DeepMind Technologies Limited.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

from enum import StrEnum


class LengthUnit(StrEnum):
    """Length unit used in system configuration.

    Attributes:
        bohr: Atomic units of length.
        angstrom: Angstrom units.
    """

    bohr = "bohr"
    angstrom = "angstrom"


# 1 Bohr = 0.52917721067 (12) x 10^{-10} m
# https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
# Note: pyscf uses a slightly older definition of 0.52917721092 angstrom.
ONE_BOHR_IN_ANGSTROM = 0.52917721067
ONE_ANGSTROM_IN_BOHR = 1 / ONE_BOHR_IN_ANGSTROM

# 1 Hartree = 627.509474 kcal/mol
# https://en.wikipedia.org/wiki/Hartree
ONE_HARTREE_IN_KCAL = 627.509474
ONE_KCAL_IN_HARTREE = 1 / ONE_HARTREE_IN_KCAL
