# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for ECP configuration utilities and solid config factories."""

import pytest

from jaqmc.utils.atomic import core_electrons_by_pp


class TestCoreElectronsByPP:
    def test_none(self):
        """No pseudopotential should remove no core electrons."""
        assert core_electrons_by_pp("Li", None) == 0

    def test_li(self):
        """Li with ccECP removes 2 core electrons (1s2)."""
        assert core_electrons_by_pp("Li", "ccecp") == 2

    def test_ph(self):
        """PH pseudopotentials remove the configured neon core."""
        assert core_electrons_by_pp("Fe", "ph") == 10


class TestRockSaltConfig:
    def test_no_ecp(self):
        """All-electron rock salt: Li(3e) + H(1e) = 4 electrons."""
        from jaqmc.app.solid.config.rock_salt import RockSaltBuilder

        cfg = RockSaltBuilder(symbol_a="Li", symbol_b="H")()
        assert cfg.pp is None
        assert cfg.electron_spins == (2, 2)
        assert cfg.atoms[0].charge == 3
        assert cfg.atoms[1].charge == 1

    def test_with_ecp(self):
        """ECP rock salt: Li(1 valence) + H(1 valence) = 2 electrons."""
        from jaqmc.app.solid.config.rock_salt import RockSaltBuilder

        cfg = RockSaltBuilder(symbol_a="Li", symbol_b="H", pp={"Li": "ccecp"})()
        assert cfg.pp == {"Li": "ccecp"}
        assert cfg.atoms[0].charge == 1
        assert cfg.atoms[1].charge == 1
        assert cfg.electron_spins == (1, 1)
        assert sorted(cfg.ecp_coefficients) == ["Li"]

    @pytest.mark.parametrize("workflow_name", ["train", "eval"])
    def test_ph_is_accepted_by_solid_workflows(self, workflow_name):
        """Solid workflows accept PH pseudopotentials via the unified `pp` key."""
        from jaqmc.app.solid.workflow import SolidEvalWorkflow, SolidTrainWorkflow

        workflow_cls = (
            SolidTrainWorkflow if workflow_name == "train" else SolidEvalWorkflow
        )
        workflow = workflow_cls.from_config(
            {
                "system": {
                    "atoms": [
                        {"symbol": "Fe", "frac_coords": [0.0, 0.0, 0.0]},
                        {"symbol": "S", "frac_coords": [0.25, 0.25, 0.25]},
                    ],
                    "lattice": {
                        "a": [0.0, 2.0, 2.0],
                        "b": [2.0, 0.0, 2.0],
                        "c": [2.0, 2.0, 0.0],
                    },
                    "pp": "ph",
                }
            }
        )
        assert not callable(workflow.system)
        assert workflow.system.ph_elements == {"Fe", "S"}
        assert workflow.system.electron_spins == (11, 11)
        assert list(workflow.estimators) == ["potential", "ph", "total"]


class TestTwoAtomChain:
    def test_no_ecp(self):
        """All-electron Li chain: 2 * 3 = 6 electrons."""
        from jaqmc.app.solid.config.two_atom_chain import TwoAtomChainBuilder

        cfg = TwoAtomChainBuilder(symbol="Li")()
        assert cfg.pp is None
        assert cfg.electron_spins == (3, 3)

    def test_with_ecp(self):
        """ECP Li chain: 2 * 1 valence = 2 electrons."""
        from jaqmc.app.solid.config.two_atom_chain import TwoAtomChainBuilder

        cfg = TwoAtomChainBuilder(symbol="Li", pp="ccecp")()
        assert cfg.pp == "ccecp"
        assert cfg.atoms[0].charge == 1
        assert cfg.atoms[1].charge == 1
        assert cfg.electron_spins == (1, 1)
        assert sorted(cfg.ecp_coefficients) == ["Li"]
