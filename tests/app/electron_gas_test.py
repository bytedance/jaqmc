# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the three-dimensional homogeneous electron gas."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.electron_gas.config import ElectronGasConfig
from jaqmc.app.electron_gas.data import data_init
from jaqmc.app.electron_gas.reference import FreeElectronReference
from jaqmc.app.electron_gas.workflow import configure_system
from jaqmc.estimator.kinetic import EuclideanKinetic, LaplacianMode
from jaqmc.utils.config import ConfigManager


def test_config_defines_wigner_seitz_volume() -> None:
    config = ElectronGasConfig(rs=2.0, nspins=(3, 2))

    expected_volume = 4 * np.pi * 5 * 2.0**3 / 3
    np.testing.assert_allclose(config.volume, expected_volume)
    np.testing.assert_allclose(np.linalg.det(config.lattice), expected_volume)

    with pytest.raises(ValueError, match="finite and positive"):
        ElectronGasConfig(rs=0.0)
    with pytest.raises(ValueError, match="At least one electron"):
        ElectronGasConfig(nspins=(0, 0))


def test_fourteen_electron_gamma_point_is_closed_shell() -> None:
    config = ElectronGasConfig(rs=1.0, nspins=(7, 7))
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
    config = ElectronGasConfig(rs=1.0, nspins=(27, 27))
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
    config = ElectronGasConfig(rs=1.0, nspins=(1, 1), twist=(0.25, 0.125, 0.0))
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
            "system": {"rs": 1.5, "nspins": [1, 1]},
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


def test_forward_laplacian_uses_smooth_periodic_features(capsys) -> None:
    """HEG kinetic energy must not materialize a full Hessian for PBC features."""
    manager = ConfigManager(
        {
            "system": {"rs": 1.0, "nspins": [1, 1]},
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
    kinetic = EuclideanKinetic(
        mode=LaplacianMode.forward_laplacian,
        f_log_psi=wavefunction.logpsi,
        data_field="electrons",
    )

    stats, _ = kinetic.evaluate_single_walker(
        params, one_walker, {}, None, jax.random.PRNGKey(2)
    )
    jax.block_until_ready(stats["energy:kinetic"])
    captured = capsys.readouterr()

    assert jnp.isfinite(stats["energy:kinetic"])
    assert "full hessian" not in (captured.out + captured.err).lower()
