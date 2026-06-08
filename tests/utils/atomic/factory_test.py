# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from jaqmc.utils.atomic import electron_spins_from_total, make_atom


def test_make_atom_all_electron_uses_default_charge():
    """All-electron atoms inherit Atom's default nuclear charge."""
    atom = make_atom("Fe", [0.0, 0.0, 0.0])

    assert atom.charge == 26


def test_make_atom_ecp_sets_valence_charge():
    """ECP atoms get charge equal to the valence electron count."""
    atom = make_atom("Li", [0.0, 0.0, 0.0], pp="ccecp")

    assert atom.charge == 1


def test_make_atom_ph_sets_valence_charge():
    """PH atoms get charge equal to the PH effective valence count."""
    atom = make_atom("Fe", [0.0, 0.0, 0.0], pp="ph")

    assert atom.charge == 16


def test_make_atom_rejects_ph_for_unsupported_element():
    """``pp="ph"`` must fail loudly for elements outside the PH library."""
    with pytest.raises(ValueError, match="SUPPORTED_PH_ELEMENTS"):
        make_atom("H", [0.0, 0.0, 0.0], pp="ph")


def test_electron_spins_from_total_respects_spin():
    """Nonzero spin biases the alpha channel by spin/2."""
    assert electron_spins_from_total(11, 1) == (6, 5)


def test_electron_spins_from_total_rejects_parity_mismatch():
    """Total electrons and spin must share parity."""
    with pytest.raises(ValueError, match="same parity"):
        electron_spins_from_total(10, 1)
