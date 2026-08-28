# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pair correlation function estimator."""

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.config import HallConfig
from jaqmc.app.hall.data import HallData, data_init
from jaqmc.app.hall.estimator.pair_correlation import PairCorrelation

KEY = jax.random.PRNGKey(0)


def test_weighted_sum_is_normalized():
    """Uniform samples give int g(theta) sin(theta) dtheta = 2 * (N - 1) / N."""
    bins = 100
    n_walkers = 1024
    n_steps = 2
    nelec = 4
    bin_width = jnp.pi / bins
    est = PairCorrelation(bins=bins)
    batched = data_init(HallConfig(nspins=(nelec, 0)), n_walkers, KEY)
    state = est.init(HallData(electrons=jnp.zeros((nelec, 2))), KEY)
    for _ in range(n_steps):
        _, state = est.evaluate_batch_walkers({}, batched, {}, state, KEY)
    g = state["histogram"][0] * 4 * bins / (jnp.pi * nelec**2 * n_walkers * n_steps)
    bin_centers = (jnp.arange(bins) + 0.5) * bin_width
    weighted_sum = jnp.sum(g * jnp.sin(bin_centers)) * bin_width
    expected = 2 * (nelec - 1) / nelec
    np.testing.assert_allclose(float(weighted_sum), float(expected), rtol=1e-2)
