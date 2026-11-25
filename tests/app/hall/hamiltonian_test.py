# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hall system: walker initialization and potential."""

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.config import HallConfig
from jaqmc.app.hall.data import HallData, data_init
from jaqmc.app.hall.hamiltonian import SpherePotential


def _evaluate(estimator, electrons, prev_walker_stats=None):
    stats, _ = estimator.evaluate_single_walker(
        {},
        HallData(electrons=electrons),
        prev_walker_stats or {},
        None,
        jax.random.key(0),
    )
    return stats


def test_data_init_samples_sphere_angles():
    cfg = HallConfig(flux=4, nspins=(2, 1))
    electrons = data_init(cfg, size=32, rngs=jax.random.key(1)).data.electrons

    assert electrons.shape == (32, 3, 2)
    theta, phi = electrons[..., 0], electrons[..., 1]
    assert jnp.all((theta >= 0) & (theta <= jnp.pi))
    assert jnp.all((phi >= -jnp.pi) & (phi <= jnp.pi))


def test_coulomb_potential_of_antipodal_pair():
    """Electrons at opposite poles are 2R apart: potential = 1 / (2R)."""
    estimator = SpherePotential(
        monopole_strength=1.0, radius=1.0, interaction_strength=1.0
    )
    stats = _evaluate(estimator, jnp.array([[0.0, 0.0], [jnp.pi, 0.0]]))
    np.testing.assert_allclose(stats["energy:potential"], 0.5, atol=1e-5)
