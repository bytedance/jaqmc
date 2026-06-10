# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import serde

from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.app.molecule.workflow import configure_system as configure_molecule_system
from jaqmc.app.molecule.workflow import make_scf as make_molecule_scf
from jaqmc.app.solid.workflow import configure_system as configure_solid_system
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR


def test_molecule_configure_system_normalizes_angstrom_before_scf():
    cfg = ConfigManager(
        {
            "system": {
                "unit": "angstrom",
                "atoms": [
                    {"symbol": "H", "coords": [0.0, 0.0, 0.74]},
                    {"symbol": "H", "coords": [0.0, 0.0, -0.74]},
                ],
                "electron_spins": [1, 1],
            }
        }
    )

    system_config, wf = configure_molecule_system(cfg)
    scf = make_molecule_scf(system_config)

    expected_coords = np.array(
        [
            [0.0, 0.0, 0.74 * ONE_ANGSTROM_IN_BOHR],
            [0.0, 0.0, -0.74 * ONE_ANGSTROM_IN_BOHR],
        ]
    )

    assert system_config.unit == "bohr"
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
                "lattice_vectors": [
                    [0.0, 2.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 2.0, 0.0],
                ],
                "atoms": [
                    {"symbol": "Li", "coords": [0.0, 0.0, 0.0]},
                    {"symbol": "H", "coords": [2.0, 2.0, 2.0]},
                ],
                "electron_spins": [2, 2],
                "supercell_matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                "basis": "sto-3g",
            }
        }
    )

    system_config, wf, scf, _ = configure_solid_system(cfg)

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

    assert system_config.unit == "bohr"
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
