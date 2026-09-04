# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pair-correlation estimator."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.pair_correlation import PairCorrelation
from jaqmc.data import BatchedData
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.utils import parallel_jax


def _make_batched(electrons: jnp.ndarray) -> BatchedData:
    return BatchedData(
        data=HallData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


def _sample_batch(batch_size: int) -> jnp.ndarray:
    electrons = jnp.array([[0.5, 0.25], [1.5, 1.25]])
    return jnp.broadcast_to(electrons, (batch_size, *electrons.shape))


class TestPairCorrelation:
    def test_digest_outputs_normalized_pair_correlation(self):
        estimator = PairCorrelation(bins=4)
        estimators = EstimatorPipeline({"pair_correlation": estimator})
        batched_data = _make_batched(_sample_batch(2))
        state = estimator.init(batched_data.unbatched_example(), jax.random.PRNGKey(0))

        assert state.shape == (jax.device_count(), estimator.bins)

        _, one_step_state = estimator.evaluate_batch_walkers(
            {}, batched_data, {}, state, jax.random.PRNGKey(1)
        )
        _, two_step_state = estimator.evaluate_batch_walkers(
            {}, batched_data, {}, one_step_state, jax.random.PRNGKey(2)
        )

        result = estimators.digest({}, {"pair_correlation": two_step_state}, n_steps=2)

        assert result.keys() == {"pair_correlation"}
        np.testing.assert_allclose(
            result["pair_correlation"], one_step_state[0], rtol=1e-6
        )

    @pytest.mark.skipif(
        jax.device_count() < 2,
        reason="requires multiple devices",
    )
    def test_sharded_state_accumulates_on_multiple_devices(self):
        estimator = PairCorrelation(bins=4)
        batch_size = 2 * jax.device_count()
        batched_data = _make_batched(_sample_batch(batch_size))
        data_sharding = parallel_jax.make_sharding(parallel_jax.DATA_PARTITION)
        batched_data = jax.device_put(batched_data, data_sharding)
        state = jax.device_put(
            estimator.init(batched_data.unbatched_example(), jax.random.PRNGKey(0)),
            data_sharding,
        )

        evaluate = parallel_jax.jit_sharded(
            lambda data, est_state: estimator.evaluate_batch_walkers(
                {}, data, {}, est_state, jax.random.PRNGKey(1)
            )[1],
            in_specs=(parallel_jax.DATA_PARTITION, parallel_jax.DATA_PARTITION),
            out_specs=parallel_jax.DATA_PARTITION,
        )

        state = evaluate(batched_data, state)
        result = estimator.finalize_state(state, n_steps=1)

        assert state.shape == (jax.device_count(), estimator.bins)
        np.testing.assert_allclose(
            result["pair_correlation"], np.asarray(state)[0], rtol=1e-6
        )
