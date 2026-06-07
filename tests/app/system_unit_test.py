# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import serde

from jaqmc.app.molecule.config.base import MoleculePretrainReferenceConfig
from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.app.molecule.workflow import configure_system as configure_molecule_system
from jaqmc.app.molecule.workflow import make_scf as make_molecule_scf
from jaqmc.app.solid.config.base import SolidAtomConfig, SolidPretrainReferenceConfig
from jaqmc.app.solid.workflow import configure_system as configure_solid_system
from jaqmc.app.solid.workflow import make_scf as make_solid_scf
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit


def test_molecule_configure_system_normalizes_angstrom_before_scf():
    cfg = ConfigManager(
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

    system_config, wf = configure_molecule_system(cfg)
    scf = make_molecule_scf(
        MoleculePretrainReferenceConfig(basis="sto-3g"), system_config
    )

    expected_coords = np.array(
        [
            [0.0, 0.0, 0.74 * ONE_ANGSTROM_IN_BOHR],
            [0.0, 0.0, -0.74 * ONE_ANGSTROM_IN_BOHR],
        ]
    )

    assert system_config.unit == LengthUnit.angstrom
    assert isinstance(wf, FermiNetWavefunction)
    assert wf.nspins == (1, 1)
    np.testing.assert_allclose(
        [atom.coords for atom in system_config.atoms],
        expected_coords,
    )
    np.testing.assert_allclose(
        scf._mol.atom_coords(unit="Bohr"),
        expected_coords,
    )


def test_solid_configure_system_normalizes_angstrom_before_scf():
    cfg = ConfigManager(
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

    system_config, wf, _ = configure_solid_system(cfg)
    scf = make_solid_scf(SolidPretrainReferenceConfig(basis="sto-3g"), system_config)

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
    np.testing.assert_allclose(np.asarray(wf.primitive_lattice), expected_lattice)
    np.testing.assert_allclose(
        np.asarray(wf.simulation_lattice),
        np.asarray(system_config.supercell_lattice),
    )
    np.testing.assert_allclose(np.asarray(scf._cell.a), expected_lattice)
    np.testing.assert_allclose(scf._cell.atom_coords(unit="Bohr"), expected_coords)


def test_system_config_rejects_unknown_length_unit():
    cfg = ConfigManager({"system": {"unit": "non-existing"}})

    with pytest.raises(serde.SerdeError, match="not a valid LengthUnit"):
        configure_molecule_system(cfg)


def test_solid_configure_system_accepts_plain_lattice_params_yaml():
    cfg = ConfigManager(
        {
            "system": {
                "unit": "angstrom",
                "lattice": {"a": 1.0, "b": 2.0, "c": 3.0},
                "atoms": [{"symbol": "He", "frac_coords": [0.0, 0.0, 0.0]}],
                "supercell_matrix": [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
            }
        }
    )

    system_config, wf, _ = configure_solid_system(cfg)

    expected_lattice = np.diag(np.array([1.0, 2.0, 3.0]) * ONE_ANGSTROM_IN_BOHR)
    expected_supercell = np.diag(np.array([1.0, 4.0, 3.0]) * ONE_ANGSTROM_IN_BOHR)

    np.testing.assert_allclose(system_config.lattice_vectors, expected_lattice)
    np.testing.assert_allclose(system_config.supercell_lattice, expected_supercell)
    np.testing.assert_allclose(
        [atom.coords for atom in system_config.atoms], [[0.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(np.asarray(wf.primitive_lattice), expected_lattice)
    np.testing.assert_allclose(np.asarray(wf.simulation_lattice), expected_supercell)


def test_solid_configure_system_rejects_cartesian_atom_coords():
    cfg = ConfigManager(
        {
            "system": {
                "lattice": {
                    "a": [0.0, 2.0, 2.0],
                    "b": [2.0, 0.0, 2.0],
                    "c": [2.0, 2.0, 0.0],
                },
                "atoms": [{"symbol": "Li", "coords": [0.0, 0.0, 0.0]}],
            }
        }
    )

    with pytest.raises(serde.SerdeError, match="coords"):
        configure_solid_system(cfg)


@pytest.mark.parametrize("frac_coords", ([1.0, 0.0, 0.0], [-0.1, 0.0, 0.0]))
def test_solid_atom_config_rejects_out_of_range_fractional_coords(frac_coords):
    with pytest.raises(ValueError, match="0 <= coord < 1"):
        SolidAtomConfig(symbol="Li", frac_coords=frac_coords)


def test_diatomic_helper_uses_s_z_convention():
    cfg = ConfigManager(
        {
            "system": {
                "module": "diatomic",
                "formula": "H2",
                "bond_length": 1.4,
                "s_z": 1,
            }
        }
    )

    system_config, wf = configure_molecule_system(cfg)

    assert system_config.electron_spins == (2, 0)
    assert isinstance(wf, FermiNetWavefunction)
    assert wf.nspins == (2, 0)
