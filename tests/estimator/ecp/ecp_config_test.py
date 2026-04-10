# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for ECP configuration utilities and solid config factories.

Covers:
- get_valence_spin_config: valence electron counting via PySCF
- get_core_electrons: core electron removal mapping
- rock_salt_config: FCC rock salt with ECP
- two_atom_chain: 1D chain with ECP
"""

from jaqmc.utils.atomic import Atom
from jaqmc.utils.atomic.ecp import get_core_electrons, get_valence_spin_config


class TestGetValenceSpinConfig:
    def test_li_ccecp(self):
        """Li with ccECP should have 1 valence electron (2 core removed)."""
        n_alpha, n_beta = get_valence_spin_config("Li", "ccecp")
        assert n_alpha + n_beta == 1
        # Li ground state: 1 unpaired electron -> all alpha
        assert n_alpha == 1
        assert n_beta == 0

    def test_per_element_dict(self):
        """Per-element ECP dict should work the same as a string for Li."""
        n_alpha, n_beta = get_valence_spin_config("Li", {"Li": "ccecp"})
        assert n_alpha == 1
        assert n_beta == 0


class TestGetCoreElectrons:
    def test_none(self):
        """No ECP should return an empty dict."""
        atoms = [Atom(symbol="Li", coords=[0.0, 0.0, 0.0])]
        result = get_core_electrons(atoms, ecp=None)
        assert result == {}

    def test_li(self):
        """Li with ccECP removes 2 core electrons (1s2)."""
        atoms = [Atom(symbol="Li", coords=[0.0, 0.0, 0.0])]
        result = get_core_electrons(atoms, ecp="ccecp")
        assert result == {"Li": 2}

    def test_per_element_dict(self):
        """Per-element ECP dict: only elements in the dict get core removal."""
        atoms = [
            Atom(symbol="Li", coords=[0.0, 0.0, 0.0]),
            Atom(symbol="H", coords=[1.0, 0.0, 0.0]),
        ]
        result = get_core_electrons(atoms, ecp={"Li": "ccecp"})
        # Li: 3 total - 1 valence = 2 core
        assert result == {"Li": 2}
        # H has no core electrons removed, so it's not in the dict
        assert "H" not in result

    def test_deduplicates(self):
        """Multiple atoms of the same element produce a single dict entry."""
        atoms = [
            Atom(symbol="Li", coords=[0.0, 0.0, 0.0]),
            Atom(symbol="Li", coords=[1.0, 0.0, 0.0]),
        ]
        result = get_core_electrons(atoms, ecp="ccecp")
        assert result == {"Li": 2}


class TestRockSaltConfig:
    def test_no_ecp(self):
        """All-electron rock salt: Li(3e) + H(1e) = 4 electrons."""
        from jaqmc.app.solid.config.rock_salt import rock_salt_config

        cfg = rock_salt_config(symbol_a="Li", symbol_b="H")
        assert cfg.ecp is None
        assert cfg.electron_spins == (2, 2)
        assert cfg.atoms[0].charge == 3  # Li all-electron
        assert cfg.atoms[1].charge == 1  # H all-electron

    def test_with_ecp(self):
        """ECP rock salt: Li(1 valence) + H(1 valence) = 2 electrons."""
        from jaqmc.app.solid.config.rock_salt import rock_salt_config

        cfg = rock_salt_config(symbol_a="Li", symbol_b="H", ecp={"Li": "ccecp"})
        assert cfg.ecp == {"Li": "ccecp"}
        # Li: 1 valence electron, H: 1 electron (no ECP)
        assert cfg.atoms[0].charge == 1
        assert cfg.atoms[1].charge == 1
        assert cfg.electron_spins == (1, 1)


class TestTwoAtomChain:
    def test_no_ecp(self):
        """All-electron Li chain: 2 * 3 = 6 electrons."""
        from jaqmc.app.solid.config.two_atom_chain import two_atom_chain

        cfg = two_atom_chain(symbol="Li")
        assert cfg.ecp is None
        assert cfg.electron_spins == (3, 3)

    def test_with_ecp(self):
        """ECP Li chain: 2 * 1 valence = 2 electrons."""
        from jaqmc.app.solid.config.two_atom_chain import two_atom_chain

        cfg = two_atom_chain(symbol="Li", ecp="ccecp")
        assert cfg.ecp == "ccecp"
        assert cfg.atoms[0].charge == 1
        assert cfg.atoms[1].charge == 1
        assert cfg.electron_spins == (1, 1)
