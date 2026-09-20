# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Haldane-sphere angular-momentum estimator."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.estimator.angular_momentum import SphericalAngularMomentum


@pytest.mark.x64_modes
@pytest.mark.parametrize("u_power", [0, 2], ids=["singlet", "highest_weight"])
@pytest.mark.parametrize(
    "first_theta",
    [
        pytest.param(None, id="generic"),
        pytest.param(np.pi / 2, id="chart_seam"),
        pytest.param(np.pi / 2 - 1e-6, id="beside_seam"),
        pytest.param(1e-3, id="near_pole"),
        pytest.param(1e-7, id="at_pole"),
    ],
)
def test_eigenstates_are_stable_near_chart_boundaries(
    x64_mode: bool,
    u_power: int,
    first_theta: float | None,
):
    """Known eigenstates retain their exact values near chart boundaries."""
    nelec = 5
    pair_power = 3
    electrons = jnp.asarray(
        [
            [1.1, 0.2],
            [2.0, -1.3],
            [0.8, 2.2],
            [2.4, 0.9],
            [1.6, -2.5],
        ],
        dtype=jnp.float64 if x64_mode else jnp.float32,
    )
    if first_theta is not None:
        electrons = electrons.at[0, 0].set(first_theta)

    def log_psi(_params, u, v):
        pair = u[:, None] * v[None, :] - v[:, None] * u[None, :]
        upper = jnp.triu(jnp.ones((nelec, nelec), dtype=bool), k=1)
        value = pair_power * jnp.sum(jnp.log(jnp.where(upper, pair, 1.0)))
        return value + u_power * jnp.sum(jnp.log(u))

    lz = nelec * u_power / 2
    estimator = SphericalAngularMomentum(
        f_log_psi_from_spinor=log_psi,
    )
    stats, _ = estimator.evaluate_single_walker(
        params={},
        data=HallData(electrons=electrons),
        prev_walker_stats={},
        state=None,
        rngs=jax.random.key(0),
    )
    atol = 1e-8 if x64_mode else 1e-4

    np.testing.assert_allclose(stats["angular_momentum_z"], lz, rtol=0, atol=atol)
    np.testing.assert_allclose(
        stats["angular_momentum_z_square"], lz**2, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        stats["angular_momentum_square"], lz * (lz + 1), rtol=0, atol=atol
    )
