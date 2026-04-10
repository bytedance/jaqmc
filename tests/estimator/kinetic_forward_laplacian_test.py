# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import dataclass

import h5py
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hydrogen_atom import hydrogen_atom_train_workflow
from jaqmc.data import Data
from jaqmc.estimator.kinetic import EuclideanKinetic, LaplacianMode
from jaqmc.utils.config import ConfigManager


@dataclass
class SimpleData(Data):
    """Simple data class for testing."""

    positions: jnp.ndarray


@pytest.mark.parametrize(
    "n_particles,n_dims,coeff",
    [
        (1, 1, 0.5),  # 1D harmonic oscillator
        (3, 3, 0.5),  # 3D few particles
        (2, 3, 0.3),  # Different coefficient
        (5, 3, 0.3),  # More particles
        (3, 2, 0.3),  # 2D
        (10, 1, 0.3),  # Many particles in 1D
        (20, 3, 0.2),  # Large system
    ],
)
def test_forward_laplacian_vs_scan_equivalence(n_particles, n_dims, coeff):
    """Test forward_laplacian matches standard across configurations."""
    pytest.importorskip("folx")
    pytest.importorskip("jax", minversion="0.7.1")

    def log_psi(params, data):
        del params
        return -coeff * jnp.sum(data["positions"] ** 2)

    key = jax.random.key(42 + n_particles * 10 + n_dims)
    positions = jax.random.normal(key, (n_particles, n_dims))
    data = SimpleData(positions=positions)
    params = {}

    estimator_scan = EuclideanKinetic(
        mode=LaplacianMode.scan, f_log_psi=log_psi, data_field="positions"
    )
    estimator_fwd_lap = EuclideanKinetic(
        mode=LaplacianMode.forward_laplacian, f_log_psi=log_psi, data_field="positions"
    )

    stats_scan, _ = estimator_scan.evaluate_local(params, data, {}, None, key)
    stats_fwd_lap, _ = estimator_fwd_lap.evaluate_local(params, data, {}, None, key)

    ke_scan = float(stats_scan["energy:kinetic"])
    ke_fwd_lap = float(stats_fwd_lap["energy:kinetic"])

    assert np.isclose(ke_scan, ke_fwd_lap, rtol=1e-5, atol=1e-6), (
        f"Config ({n_particles}p, {n_dims}D, c={coeff}): "
        f"standard={ke_scan}, forward_laplacian={ke_fwd_lap}"
    )


def test_forward_laplacian_kinetic_with_complex_wavefunction():
    """Test forward_laplacian kinetic with a complex-valued wavefunction."""
    pytest.importorskip("folx")
    pytest.importorskip("jax", minversion="0.7.1")

    # Complex wavefunction: psi = exp(i*k*x) * exp(-x^2/2)
    # log(psi) = i*k*x - x^2/2
    def log_psi_complex(params, data):
        del params
        positions = data["positions"]
        k = 1.0  # wave vector
        return 1j * k * jnp.sum(positions) - 0.5 * jnp.sum(positions**2)

    key = jax.random.key(777)
    positions = jax.random.normal(key, (2, 3))
    data = SimpleData(positions=positions)
    params = {}

    estimator_fwd_lap = EuclideanKinetic(
        mode=LaplacianMode.forward_laplacian,
        f_log_psi=log_psi_complex,
        data_field="positions",
    )
    estimator_scan = EuclideanKinetic(
        mode=LaplacianMode.scan, f_log_psi=log_psi_complex, data_field="positions"
    )

    stats_fwd_lap, _ = estimator_fwd_lap.evaluate_local(params, data, {}, None, key)
    stats_scan, _ = estimator_scan.evaluate_local(params, data, {}, None, key)

    # Kinetic energy can be complex for complex wavefunctions
    ke_fwd_lap = stats_fwd_lap["energy:kinetic"]
    ke_scan = stats_scan["energy:kinetic"]

    # Extract real and imaginary parts
    ke_fwd_lap_real = float(jnp.real(ke_fwd_lap))
    ke_fwd_lap_imag = float(jnp.imag(ke_fwd_lap))
    ke_scan_real = float(jnp.real(ke_scan))
    ke_scan_imag = float(jnp.imag(ke_scan))

    # Both real and imaginary parts should match
    assert np.isclose(ke_fwd_lap_real, ke_scan_real, rtol=1e-5, atol=1e-6), (
        f"Real part mismatch: fwd_lap={ke_fwd_lap_real}, standard={ke_scan_real}"
    )
    assert np.isclose(ke_fwd_lap_imag, ke_scan_imag, rtol=1e-5, atol=1e-6), (
        f"Imag part mismatch: fwd_lap={ke_fwd_lap_imag}, standard={ke_scan_imag}"
    )


def test_forward_laplacian_kinetic_edge_cases():
    """Test edge cases like zero positions and extreme values."""
    pytest.importorskip("folx")
    pytest.importorskip("jax", minversion="0.7.1")

    def log_psi(params, data):
        del params
        return -0.5 * jnp.sum(data["positions"] ** 2)

    params = {}
    key = jax.random.key(42)

    estimator_fwd_lap = EuclideanKinetic(
        mode=LaplacianMode.forward_laplacian,
        f_log_psi=log_psi,
        data_field="positions",
    )
    estimator_scan = EuclideanKinetic(
        mode=LaplacianMode.scan, f_log_psi=log_psi, data_field="positions"
    )

    # Test zero positions
    data = SimpleData(positions=jnp.zeros((3, 3)))
    stats_fwd_lap, _ = estimator_fwd_lap.evaluate_local(params, data, {}, None, key)
    stats_scan, _ = estimator_scan.evaluate_local(params, data, {}, None, key)

    ke_fwd_lap = float(stats_fwd_lap["energy:kinetic"])
    ke_scan = float(stats_scan["energy:kinetic"])

    assert jnp.isfinite(ke_fwd_lap), (
        "forward_laplacian kinetic energy should be finite for zeros"
    )
    assert jnp.isfinite(ke_scan), "Standard kinetic energy should be finite for zeros"
    assert np.isclose(ke_fwd_lap, ke_scan, rtol=1e-5, atol=1e-6), (
        f"Zero positions: forward_laplacian={ke_fwd_lap}, standard={ke_scan}"
    )

    # Test very small positions
    data = SimpleData(positions=jnp.ones((3, 3)) * 1e-10)
    stats_fwd_lap, _ = estimator_fwd_lap.evaluate_local(params, data, {}, None, key)
    stats_scan, _ = estimator_scan.evaluate_local(params, data, {}, None, key)

    ke_fwd_lap = float(stats_fwd_lap["energy:kinetic"])
    ke_scan = float(stats_scan["energy:kinetic"])

    assert jnp.isfinite(ke_fwd_lap), (
        "forward_laplacian kinetic energy should be finite for small values"
    )
    assert jnp.isfinite(ke_scan), (
        "Standard kinetic energy should be finite for small values"
    )
    assert np.isclose(ke_fwd_lap, ke_scan, rtol=1e-5, atol=1e-6), (
        f"Small positions: forward_laplacian={ke_fwd_lap}, standard={ke_scan}"
    )

    # Test single particle, single dimension
    data = SimpleData(positions=jnp.array([[0.5]]))
    stats_fwd_lap, _ = estimator_fwd_lap.evaluate_local(params, data, {}, None, key)
    stats_scan, _ = estimator_scan.evaluate_local(params, data, {}, None, key)

    ke_fwd_lap = float(stats_fwd_lap["energy:kinetic"])
    ke_scan = float(stats_scan["energy:kinetic"])

    assert jnp.isfinite(ke_fwd_lap), (
        "forward_laplacian kinetic energy should be finite for single element"
    )
    assert jnp.isfinite(ke_scan), (
        "Standard kinetic energy should be finite for single element"
    )
    assert np.isclose(ke_fwd_lap, ke_scan, rtol=1e-5, atol=1e-6), (
        f"Single element: forward_laplacian={ke_fwd_lap}, standard={ke_scan}"
    )


def test_forward_laplacian_vs_scan_molecule_workflow(tmp_path):
    """Compare default kinetic mode against scan in actual molecule workflow."""
    pytest.importorskip("folx")
    pytest.importorskip("jax", minversion="0.7.1")

    base_config = {
        "workflow": {"seed": 42},
        "train": {
            "optim": {"learning_rate": {"rate": 0.02}},
            "run": {"burn_in": 5, "iterations": 10},
        },
    }

    # Run with scan
    config_scan = copy.deepcopy(base_config)
    config_scan["workflow"]["save_path"] = str(tmp_path / "scan")
    config_scan["energy"] = {"kinetic": {"mode": "scan"}}
    cfg_scan = ConfigManager(config_scan)
    hydrogen_atom_train_workflow(cfg_scan)()

    # Run with forward_laplacian
    config_fwd_lap = copy.deepcopy(base_config)
    config_fwd_lap["workflow"]["save_path"] = str(tmp_path / "forward_laplacian")
    config_fwd_lap["energy"] = {"kinetic": {"mode": "forward_laplacian"}}
    cfg_fwd_lap = ConfigManager(config_fwd_lap)
    hydrogen_atom_train_workflow(cfg_fwd_lap)()

    # Compare results
    with h5py.File(tmp_path / "scan" / "train_stats.h5", "r") as f_std:
        ke_scan = f_std["energy:kinetic"][:]
        loss_scan = f_std["loss"][:]

    with h5py.File(tmp_path / "forward_laplacian" / "train_stats.h5", "r") as f_fwd_lap:
        ke_fwd_lap = f_fwd_lap["energy:kinetic"][:]
        loss_fwd_lap = f_fwd_lap["loss"][:]

    # Kinetic energies should be very similar across all iterations
    # (not necessarily identical due to different compilation/execution paths)
    for i in range(len(ke_scan)):
        assert np.isclose(ke_scan[i], ke_fwd_lap[i], rtol=1e-3, atol=1e-4), (
            f"Iteration {i}: scan={ke_scan[i]}, default={ke_fwd_lap[i]}"
        )

    # Final losses should also be similar
    assert np.isclose(loss_scan[-1], loss_fwd_lap[-1], rtol=1e-2), (
        f"Final loss differs: scan={loss_scan[-1]}, default={loss_fwd_lap[-1]}"
    )
