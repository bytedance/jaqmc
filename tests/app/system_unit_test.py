# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from jaqmc.app.molecule.config import MoleculeConfig
from jaqmc.app.molecule.workflow import MoleculeTrainWorkflow
from jaqmc.app.solid.config.base import (
    LatticeParameters,
    SolidAtomConfig,
    SolidConfig,
)
from jaqmc.app.solid.workflow import SolidTrainWorkflow
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit


def test_molecule_system_normalizes_angstrom():
    workflow = MoleculeTrainWorkflow.from_config(
        {
            "system": {
                "unit": "angstrom",
                "atoms": [
                    {"symbol": "H", "coords": [0.0, 0.0, 0.74]},
                    {"symbol": "H", "coords": [0.0, 0.0, -0.74]},
                ],
            }
        }
    )

    assert isinstance(workflow.system, MoleculeConfig)
    system_config = workflow.system

    expected_coords = np.array(
        [
            [0.0, 0.0, 0.74 * ONE_ANGSTROM_IN_BOHR],
            [0.0, 0.0, -0.74 * ONE_ANGSTROM_IN_BOHR],
        ]
    )

    assert system_config.unit == LengthUnit.angstrom
    assert system_config.electron_spins == (1, 1)
    np.testing.assert_allclose(
        [atom.coords for atom in system_config.atoms],
        expected_coords,
    )


def test_solid_system_normalizes_angstrom():
    workflow = SolidTrainWorkflow.from_config(
        {
            "system": {
                "unit": "angstrom",
                "lattice": {
                    "a": [0.0, 2.0, 2.0],
                    "b": [2.0, 0.0, 2.0],
                    "c": [2.0, 2.0, 0.0],
                },
                "atoms": [
                    {"symbol": "Li", "frac_coords": [0.0, 0.0, 0.0]},
                    {"symbol": "H", "frac_coords": [0.5, 0.5, 0.5]},
                ],
                "supercell_matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        }
    )

    assert isinstance(workflow.system, SolidConfig)
    system_config = workflow.system

    expected_lattice = np.array(
        [
            [0.0, 2.0 * ONE_ANGSTROM_IN_BOHR, 2.0 * ONE_ANGSTROM_IN_BOHR],
            [2.0 * ONE_ANGSTROM_IN_BOHR, 0.0, 2.0 * ONE_ANGSTROM_IN_BOHR],
            [2.0 * ONE_ANGSTROM_IN_BOHR, 2.0 * ONE_ANGSTROM_IN_BOHR, 0.0],
        ]
    )
    expected_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [
                2.0 * ONE_ANGSTROM_IN_BOHR,
                2.0 * ONE_ANGSTROM_IN_BOHR,
                2.0 * ONE_ANGSTROM_IN_BOHR,
            ],
        ]
    )

    assert system_config.unit == LengthUnit.angstrom
    np.testing.assert_allclose(system_config.lattice_vectors, expected_lattice)
    np.testing.assert_allclose(
        system_config.supercell_lattice,
        np.dot(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]), expected_lattice),
    )
    np.testing.assert_allclose(
        [atom.coords for atom in system_config.atoms],
        expected_coords,
    )


def test_solid_config_rejects_singular_supercell_matrix():
    with pytest.raises(ValueError, match="non-singular"):
        SolidConfig(supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])


def test_system_config_rejects_unknown_length_unit():
    with pytest.raises(ValidationError, match="unit"):
        MoleculeTrainWorkflow.from_config({"system": {"unit": "non-existing"}})


def test_solid_system_accepts_lattice_params_yaml():
    workflow = SolidTrainWorkflow.from_config(
        {
            "system": {
                "unit": "angstrom",
                "lattice": {
                    "a": 1.0,
                    "b": 2.0,
                    "c": 3.0,
                },
                "atoms": [{"symbol": "He", "frac_coords": [0.0, 0.0, 0.0]}],
                "supercell_matrix": [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        }
    )

    assert isinstance(workflow.system, SolidConfig)
    system_config = workflow.system

    expected_lattice = np.diag(np.array([1.0, 2.0, 3.0]) * ONE_ANGSTROM_IN_BOHR)
    expected_supercell = np.diag(np.array([1.0, 4.0, 3.0]) * ONE_ANGSTROM_IN_BOHR)

    np.testing.assert_allclose(system_config.lattice_vectors, expected_lattice)
    np.testing.assert_allclose(system_config.supercell_lattice, expected_supercell)
    np.testing.assert_allclose(
        [atom.coords for atom in system_config.atoms], [[0.0, 0.0, 0.0]]
    )


def test_solid_lattice_merges_partial_layers_into_the_default_variant():
    untagged = SolidTrainWorkflow.from_config(
        {"system": {"lattice": {"a": 1.0, "b": 2.0, "c": 3.0}}}
    )
    partial = SolidTrainWorkflow.from_config({"system": {"lattice": {"a": 4.0}}})

    assert isinstance(untagged.system, SolidConfig)
    assert isinstance(partial.system, SolidConfig)
    assert untagged.system.lattice == LatticeParameters(a=1.0, b=2.0, c=3.0)
    assert partial.system.lattice == LatticeParameters(a=4.0, b=2, c=2)


def test_solid_lattice_switch_requires_the_selected_variant_fields():
    with pytest.raises(ValidationError, match=r"lattice|LatticeVectors"):
        SolidTrainWorkflow.from_config({"system": {"lattice": {"a": [1.0, 0.0, 0.0]}}})


def test_solid_lattice_round_trips_through_yaml_without_class_tag():
    workflow = SolidTrainWorkflow.from_config({})

    serialized = workflow.to_yaml()
    restored = SolidTrainWorkflow.from_config(yaml.safe_load(serialized))

    assert "class:" not in serialized
    assert restored.system == workflow.system


def test_solid_configure_system_rejects_cartesian_atom_coords():
    config = {
        "system": {
            "lattice": {
                "a": [0.0, 2.0, 2.0],
                "b": [2.0, 0.0, 2.0],
                "c": [2.0, 2.0, 0.0],
            },
            "atoms": [{"symbol": "Li", "coords": [0.0, 0.0, 0.0]}],
        }
    }

    with pytest.raises(ValidationError, match="coords"):
        SolidTrainWorkflow.from_config(config)


@pytest.mark.parametrize("frac_coords", ([1.0, 0.0, 0.0], [-0.1, 0.0, 0.0]))
def test_solid_atom_config_rejects_out_of_range_fractional_coords(frac_coords):
    with pytest.raises(ValueError, match="0 <= coord < 1"):
        SolidAtomConfig(symbol="Li", frac_coords=frac_coords)


def test_diatomic_helper_uses_s_z_convention():
    workflow = MoleculeTrainWorkflow.from_config(
        {
            "system": {
                "module": "diatomic",
                "formula": "H2",
                "bond_length": 1.4,
                "s_z": 1,
            }
        }
    )

    assert callable(workflow.system)
    system_config = workflow.system()

    assert system_config.electron_spins == (2, 0)


def test_molecule_builders_validate_inputs_on_construction():
    from jaqmc.app.molecule.config.alkane import AlkaneBuilder
    from jaqmc.app.molecule.config.atom import AtomBuilder
    from jaqmc.app.molecule.config.diatomic import DiatomicBuilder

    with pytest.raises(ValueError, match="Unknown element symbol"):
        AtomBuilder(symbol="unknown")
    with pytest.raises(ValueError, match="Cannot parse"):
        DiatomicBuilder(formula="H3")
    with pytest.raises(ValueError, match="repeat_num must be at least 1"):
        AlkaneBuilder(repeat_num=0)


def test_system_contracts_reject_non_system_modules():
    with pytest.raises(ValidationError, match="MoleculeConfig"):
        MoleculeTrainWorkflow.from_config(
            {"system": {"module": "base:MoleculeSolverConfig"}}
        )
    with pytest.raises(ValidationError, match="SolidConfig"):
        SolidTrainWorkflow.from_config({"system": {"module": "pyscf"}})
