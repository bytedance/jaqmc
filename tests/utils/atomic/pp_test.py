# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from jaqmc.app.molecule.config import AtomConfig, MoleculeConfig
from jaqmc.app.molecule.config.diatomic import diatomic_config
from jaqmc.utils.atomic import PH_SURROGATE_ECP, PP_PH, core_electrons_by_pp


def test_core_electrons_by_pp_routes_all_electron_ecp_and_ph():
    assert core_electrons_by_pp("H", None) == 0
    assert core_electrons_by_pp("Li", "ccecp") == 2
    assert core_electrons_by_pp("Fe", PP_PH) == 10


def test_core_electrons_by_pp_rejects_unsupported_ph_symbol():
    with pytest.raises(ValueError, match="PH is not available for element O"):
        core_electrons_by_pp("O", PP_PH)


def test_core_electrons_by_pp_rejects_unknown_ecp_name():
    with pytest.raises(
        ValueError, match="ECP 'definitely-not-an-ecp' is not supported for element Li"
    ):
        core_electrons_by_pp("Li", "definitely-not-an-ecp")


def test_atomic_system_config_mixes_ph_ecp_and_all_electron():
    """Protect integrated PH/ECP/all-electron routing and SCF surrogate ECP."""
    cfg = MoleculeConfig(
        atom_configs=[
            AtomConfig(symbol="Fe", coords=[0.0, 0.0, 0.0]),
            AtomConfig(symbol="Li", coords=[1.0, 0.0, 0.0]),
            AtomConfig(symbol="H", coords=[2.0, 0.0, 0.0]),
        ],
        pp={"Fe": "ph", "Li": "ccecp"},
        s_z=0,
    )

    assert [atom.charge for atom in cfg.atoms] == [16, 1, 1]
    assert cfg.ph_elements == {"Fe"}
    assert sorted(cfg.ecp_coefficients) == ["Li"]
    assert PH_SURROGATE_ECP["Fe"] == "ccecp"


def test_atomic_system_config_rejects_unsupported_ph_symbol():
    with pytest.raises(ValueError, match="PH is not available for element O"):
        MoleculeConfig(
            atom_configs=[AtomConfig(symbol="O", coords=[0.0, 0.0, 0.0])],
            pp={"O": "ph"},
        )


def test_atomic_system_config_rejects_unused_pp_mapping_entries():
    with pytest.raises(ValueError, match="not used"):
        MoleculeConfig(
            atom_configs=[AtomConfig(symbol="Fe", coords=[0.0, 0.0, 0.0])],
            pp={"Fe": "ph", "Q": "ph"},
            s_z=2,
        )


def test_atomic_system_config_rejects_string_ph_with_unsupported_atom():
    with pytest.raises(ValueError, match="PH is not available for element H"):
        MoleculeConfig(
            atom_configs=[
                AtomConfig(symbol="Fe", coords=[0.0, 0.0, 0.0]),
                AtomConfig(symbol="H", coords=[1.0, 0.0, 0.0]),
            ],
            pp="ph",
            s_z=8.5,
        )


def test_atomic_system_config_supports_ph_only_without_runtime_ecp():
    cfg = MoleculeConfig(
        atom_configs=[
            AtomConfig(symbol="Fe", coords=[0.0, 0.0, 0.0]),
            AtomConfig(symbol="S", coords=[1.0, 0.0, 0.0]),
        ],
        pp={"Fe": "ph", "S": "ph"},
        s_z=0,
    )

    assert cfg.ph_elements == {"Fe", "S"}
    assert cfg.ecp_coefficients == {}


def test_diatomic_config_supports_ph_treatment():
    """Guard PH valence charge propagation in mixed molecule config factories."""
    cfg = diatomic_config(formula="FeH", pp={"Fe": "ph"}, s_z=0.5)

    assert cfg.pp == {"Fe": "ph"}
    assert cfg.atoms[0].charge == 16
    assert cfg.atoms[1].charge == 1
    assert sum(cfg.electron_spins) == 17
