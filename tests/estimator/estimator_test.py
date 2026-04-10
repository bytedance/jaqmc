# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from operator import itemgetter

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.estimator import Estimator, EstimatorPipeline, FunctionEstimator


def _dummy_evaluate(params, data, stats, state, rngs):
    return {"value": 1.0}, state


class _BatchEstimator(Estimator):
    """Estimator that overrides evaluate_batch to skip vmap."""

    def __init__(self, key: str, value: float):
        self._key = key
        self._value = value

    def evaluate_batch(self, params, batched_data, prev_local_stats, state, rngs):
        return {self._key: jnp.array(self._value)}, state

    def reduce(self, local_stats):
        return local_stats

    def finalize_stats(self, batch_stats, state):
        mean_stats = jax.tree.map(lambda x: jnp.nanmean(x, axis=0), batch_stats)
        return {k: v * 2 for k, v in mean_stats.items()}


def test_function_estimator_delegates_to_fn():
    est = FunctionEstimator(_dummy_evaluate)
    result, state = est.evaluate_local(None, None, {}, "s", None)
    assert result == {"value": 1.0}
    assert state == "s"


def _make_pipeline_and_state():
    pipeline = EstimatorPipeline(
        {
            "a": _BatchEstimator("x", 1.0),
            "b": _BatchEstimator("y", 2.0),
        }
    )
    # _BatchEstimator.init returns None, so build the state dict directly
    # to avoid needing real BatchedData.
    state = {name: None for name in pipeline.estimators}
    return pipeline, state


def test_pipeline_evaluate_returns_flat_stats():
    pipeline, state = _make_pipeline_and_state()
    rngs = jax.random.PRNGKey(0)
    step_stats, _ = pipeline.evaluate(None, None, state, rngs)
    assert set(step_stats.keys()) == {"x", "y"}
    assert step_stats["x"] == pytest.approx(1.0)
    assert step_stats["y"] == pytest.approx(2.0)


def test_pipeline_finalize_stats_averages_and_dispatches():
    pipeline, state = _make_pipeline_and_state()
    rngs = jax.random.PRNGKey(0)
    step_stats, state = pipeline.evaluate(None, None, state, rngs)
    # Add batch dim of 1, same as VMC does
    batched = jax.tree.map(itemgetter(None), step_stats)
    final = pipeline.finalize_stats(batched, state)
    # _BatchEstimator.finalize_stats doubles values; averaging (1,) → scalar is identity
    assert final == {"x": 2.0, "y": 4.0}


def test_pipeline_finalize_stats_averages_stacked():
    pipeline, state = _make_pipeline_and_state()
    rngs = jax.random.PRNGKey(0)
    pipeline.evaluate(None, None, state, rngs)
    # Simulate stacked evaluation data (3 steps)
    stacked = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 6.0, 8.0])}
    final = pipeline.finalize_stats(stacked, state)
    assert final["x"].shape == ()
    assert final["y"].shape == ()
    # Pipeline averages axis 0 → 2.0/6.0, then finalize_stats doubles
    assert np.isclose(final["x"], 4.0)
    assert np.isclose(final["y"], 12.0)


def test_pipeline_finalize_stats_raises_before_evaluate():
    pipeline = EstimatorPipeline({"a": _BatchEstimator("x", 1.0)})
    with pytest.raises(RuntimeError, match=r"finalize_stats.*before evaluate"):
        pipeline.finalize_stats({"x": np.array([1.0])}, {"a": None})
