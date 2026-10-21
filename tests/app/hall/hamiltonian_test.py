# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hall system: walker initialization, potential, and penalized loss."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.config import HallConfig
from jaqmc.app.hall.data import HallData, data_init
from jaqmc.app.hall.estimator.penalized_loss import PenalizedLoss
from jaqmc.app.hall.hamiltonian import SpherePotential


def _evaluate(estimator, electrons, prev_walker_stats=None):
    stats, _ = estimator.evaluate_single_walker(
        {},
        HallData(electrons=electrons),
        prev_walker_stats or {},
        None,
        jax.random.PRNGKey(0),
    )
    return stats


def test_data_init_samples_sphere_angles():
    cfg = HallConfig(flux=4, nspins=(2, 1))
    electrons = data_init(cfg, size=32, rngs=jax.random.PRNGKey(1)).data.electrons

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


@pytest.mark.parametrize(
    "penalties,stats,expected",
    [
        pytest.param(
            {"lz_penalty": 0.0, "l2_penalty": 0.0},
            {"total_energy": 5.0},
            5.0,
            id="no-penalty",
        ),
        pytest.param(
            # 10 + 2 * (<Lz^2> - 2 * 1 * <Lz> + 1^2) = 10 + 2 * 4
            {"lz_center": 1.0, "lz_penalty": 2.0, "l2_penalty": 0.0},
            {
                "total_energy": 10.0,
                "angular_momentum_z": 3.0,
                "angular_momentum_z_square": 9.0,
            },
            18.0,
            id="lz-only",
        ),
        pytest.param(
            # 1 + 1 * <Lz^2> + 0.5 * <L^2> = 1 + 4 + 3
            {"lz_center": 0.0, "lz_penalty": 1.0, "l2_penalty": 0.5},
            {
                "total_energy": 1.0,
                "angular_momentum_z": 2.0,
                "angular_momentum_z_square": 4.0,
                "angular_momentum_square": 6.0,
            },
            8.0,
            id="lz-and-l2",
        ),
    ],
)
def test_penalized_loss(penalties, stats, expected):
    out = _evaluate(PenalizedLoss(**penalties), jnp.zeros((1, 2)), stats)
    np.testing.assert_allclose(out["penalized_loss"], expected)
