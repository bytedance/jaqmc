# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from jaqmc.app.moire.config import MoireConfig
from jaqmc.utils.units import ONE_HARTREE_IN_MEV


def test_moire_config_derives_supercell_and_unit_conversions():
    config = MoireConfig(
        electron_spins=(3, 1),
        nelec=4,
        lattice_vectors=[[1.5, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        moire_lattice_vectors=[[np.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]],
        supercell_matrix=[[3, -1], [1, 2]],
        twist=[0.1, 0.2],
        effective_mass=0.8,
        dielectric_constant=4.0,
        v1_mev=1.5,
        phi1_deg=45.0,
        omega1_mev=2.0,
        moire_lattice_constant_nm=100.0,
    )
    length_scale = config.moire_lattice_constant_bohr

    assert config.scale == 7
    np.testing.assert_allclose(
        config.supercell_lattice,
        np.dot([[3, -1], [1, 2]], [[1.5, 0.0], [0.5, np.sqrt(3.0) / 2.0]]),
    )
    assert config.moire_lattice_vectors is not None
    np.testing.assert_allclose(
        config.moire_lattice_vectors,
        [[np.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]],
    )
    np.testing.assert_allclose(config.phi1_rad, np.pi / 4.0)
    np.testing.assert_allclose(
        config.kinetic_prefactor_mev, ONE_HARTREE_IN_MEV / (length_scale**2)
    )
    np.testing.assert_allclose(
        config.coulomb_prefactor_mev, ONE_HARTREE_IN_MEV / (4.0 * length_scale)
    )


def test_moire_lattice_defaults_to_validated_lattice_vectors():
    lattice = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    config = MoireConfig(lattice_vectors=lattice)

    np.testing.assert_allclose(config.lattice_vectors, lattice)
    assert config.moire_lattice_vectors is not None
    np.testing.assert_allclose(config.moire_lattice_vectors, lattice)


def test_moire_config_rejects_non_unit_norm_moire_lattice():
    with pytest.raises(ValueError, match="unit-norm"):
        MoireConfig(moire_lattice_vectors=[[2.0, 0.0], [0.0, 1.0]])


def test_moire_config_rejects_bad_nelec():
    with pytest.raises(ValueError, match="nelec must equal sum"):
        MoireConfig(electron_spins=(2, 1), nelec=4)


def test_moire_config_rejects_bad_shapes():
    with pytest.raises(ValueError, match="lattice_vectors must have shape"):
        MoireConfig(lattice_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="supercell_matrix must have shape"):
        MoireConfig(supercell_matrix=[[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="supercell_matrix must have integer entries"):
        MoireConfig(supercell_matrix=[[2.5, 0], [0, 2]])  # type: ignore[list-item]
