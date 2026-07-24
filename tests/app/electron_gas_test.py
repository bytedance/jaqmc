# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the three-dimensional homogeneous electron gas."""

import dataclasses

import jax
import numpy as np
import pytest
import serde
from jax import numpy as jnp

from jaqmc.app.electron_gas.config import ElectronGasConfig
from jaqmc.app.electron_gas.data import data_init
from jaqmc.app.electron_gas.reference import FreeElectronReference
from jaqmc.app.electron_gas.workflow import configure_system
from jaqmc.estimator.kinetic import EuclideanKinetic, LaplacianMode
from jaqmc.utils.config import ConfigManager


def test_config_defines_wigner_seitz_volume() -> None:
    config = ElectronGasConfig(rs=2.0, nelectrons=5, s_z=0.5)

    expected_volume = 4 * np.pi * 5 * 2.0**3 / 3
    assert config.nspins == (3, 2)
    np.testing.assert_allclose(config.volume, expected_volume)
    np.testing.assert_allclose(np.linalg.det(config.lattice), expected_volume)

    with pytest.raises(ValueError, match="finite and positive"):
        ElectronGasConfig(rs=0.0, nelectrons=2, s_z=0)
    with pytest.raises(ValueError, match="positive integer"):
        ElectronGasConfig(rs=1.0, nelectrons=0, s_z=0)
    with pytest.raises(ValueError, match="finite half integer"):
        ElectronGasConfig(rs=1.0, nelectrons=2, s_z=0.25)
    with pytest.raises(ValueError, match="Impossible"):
        ElectronGasConfig(rs=1.0, nelectrons=2, s_z=0.5)

    manager = ConfigManager({"system": {"rs": 1.0, "nelectrons": 14.5, "s_z": 0}})
    with pytest.raises(serde.SerdeError, match="positive integer"):
        manager.get_module("system", "jaqmc.app.electron_gas.config")


def test_fourteen_electron_gamma_point_is_closed_shell() -> None:
    config = ElectronGasConfig(rs=1.0, nelectrons=14, s_z=0)
    reference = FreeElectronReference(config.nspins, config.lattice, config.twist)

    fractional_k = (
        np.asarray(reference.spin_kpoints[0]) * config.side_length / (2 * np.pi)
    )
    expected_shell = {
        (0, 0, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    }
    assert {tuple(np.rint(k).astype(int)) for k in fractional_k} == expected_shell
    np.testing.assert_allclose(reference.spin_kpoints[0], reference.spin_kpoints[1])
    assert reference.get_orbital_kpoints().shape == (14, 3)


def test_fifty_four_electron_gamma_point_fills_n_squared_through_three() -> None:
    config = ElectronGasConfig(rs=1.0, nelectrons=54, s_z=0)
    reference = FreeElectronReference(config.nspins, config.lattice, config.twist)

    fractional_k = (
        np.asarray(reference.spin_kpoints[0]) * config.side_length / (2 * np.pi)
    )
    integer_k = np.rint(fractional_k).astype(int)

    assert integer_k.shape == (27, 3)
    assert set(np.sum(integer_k**2, axis=1)) == {0, 1, 2, 3}
    np.testing.assert_allclose(fractional_k, integer_k)
    np.testing.assert_allclose(reference.spin_kpoints[0], reference.spin_kpoints[1])
    assert reference.get_orbital_kpoints().shape == (54, 3)


def test_free_electron_reference_obeys_twisted_boundary() -> None:
    config = ElectronGasConfig(rs=1.0, nelectrons=2, s_z=0, twist=(0.25, 0.125, 0.0))
    reference = FreeElectronReference(config.nspins, config.lattice, config.twist)
    electrons = jnp.asarray([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]])

    logpsi = reference.eval_slater(electrons, config.nspins)
    translated = electrons.at[0].add(jnp.asarray(config.lattice[0]))
    translated_logpsi = reference.eval_slater(translated, config.nspins)

    phase_ratio = jnp.exp(1j * (translated_logpsi.imag - logpsi.imag))
    expected_phase = jnp.exp(2j * jnp.pi * config.twist[0])
    assert jnp.allclose(translated_logpsi.real, logpsi.real, atol=1e-6)
    assert jnp.allclose(phase_ratio, expected_phase, atol=1e-6)


def test_uniform_data_and_solid_wavefunction_reuse() -> None:
    manager = ConfigManager(
        {
            "system": {
                "rs": 1.5,
                "nelectrons": 2,
                "s_z": 0,
                "twist": [0.25, 0.0, 0.0],
            },
            "wf": {
                "hidden_dims_single": [8],
                "hidden_dims_double": [4],
                "ndets": 1,
            },
        }
    )
    config, wavefunction, _, _ = configure_system(manager)
    batched = data_init(config, size=4, rngs=jax.random.PRNGKey(0))

    assert batched.data.electrons.shape == (4, 2, 3)
    assert batched.data.atoms.shape == (0, 3)
    fractional = batched.data.electrons @ jnp.linalg.inv(jnp.asarray(config.lattice))
    assert jnp.all((fractional >= 0) & (fractional < 1))

    one_walker = dataclasses.replace(batched.data, electrons=batched.data.electrons[0])
    params = wavefunction.init_params(one_walker, jax.random.PRNGKey(1))
    output = wavefunction.evaluate(params, one_walker)
    assert jnp.isfinite(output["logpsi"])

    translated = dataclasses.replace(
        one_walker,
        electrons=one_walker.electrons.at[0].add(jnp.asarray(config.lattice[0])),
    )
    translated_logpsi = wavefunction.logpsi(params, translated)
    phase_ratio = jnp.exp(1j * (translated_logpsi.imag - output["logpsi"].imag))
    expected_phase = jnp.exp(2j * jnp.pi * config.twist[0])
    assert jnp.allclose(translated_logpsi.real, output["logpsi"].real, atol=1e-6)
    assert jnp.allclose(phase_ratio, expected_phase, atol=1e-6)


def test_electron_gas_rejects_block_determinants() -> None:
    manager = ConfigManager(
        {
            "system": {"rs": 1.0, "nelectrons": 2, "s_z": 0},
            "wf": {"full_det": False},
        }
    )

    with pytest.raises(ValueError, match="full_det=True"):
        configure_system(manager)


def test_sparse_forward_laplacian_matches_dense_and_scan() -> None:
    manager = ConfigManager(
        {
            "system": {
                "rs": 1.0,
                "nelectrons": 2,
                "s_z": 0,
                "twist": [0.25, 0.125, 0.0],
            },
            "wf": {
                "hidden_dims_single": [8],
                "hidden_dims_double": [4],
                "ndets": 1,
            },
        }
    )
    config, wavefunction, _, _ = configure_system(manager)
    batched = data_init(config, size=1, rngs=jax.random.PRNGKey(0))
    one_walker = dataclasses.replace(batched.data, electrons=batched.data.electrons[0])
    params = wavefunction.init_params(one_walker, jax.random.PRNGKey(1))
    sparse_forward = EuclideanKinetic(
        mode=LaplacianMode.forward_laplacian,
        f_log_psi=wavefunction.logpsi,
        data_field="electrons",
        sparse=True,
    )
    dense_forward = EuclideanKinetic(
        mode=LaplacianMode.forward_laplacian,
        f_log_psi=wavefunction.logpsi,
        data_field="electrons",
        sparse=False,
    )
    scan = EuclideanKinetic(
        mode=LaplacianMode.scan,
        f_log_psi=wavefunction.logpsi,
        data_field="electrons",
    )

    sparse_stats, _ = sparse_forward.evaluate_single_walker(
        params, one_walker, {}, None, jax.random.PRNGKey(2)
    )
    dense_stats, _ = dense_forward.evaluate_single_walker(
        params, one_walker, {}, None, jax.random.PRNGKey(2)
    )
    scan_stats, _ = scan.evaluate_single_walker(
        params, one_walker, {}, None, jax.random.PRNGKey(2)
    )
    sparse_energy = sparse_stats["energy:kinetic"]
    dense_energy = dense_stats["energy:kinetic"]
    scan_energy = scan_stats["energy:kinetic"]

    assert jnp.all(
        jnp.isfinite(jnp.asarray([sparse_energy, dense_energy, scan_energy]))
    )
    np.testing.assert_allclose(sparse_energy, dense_energy, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_energy, scan_energy, rtol=1e-5, atol=1e-5)
