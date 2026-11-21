# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from operator import itemgetter
from typing import Annotated, Any, cast

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from pydantic import ValidationError

from jaqmc.data import BatchedData, Data
from jaqmc.estimator import (
    Estimator,
    EstimatorCollection,
    EstimatorPipeline,
    PerWalkerEstimator,
    WithToggles,
)
from jaqmc.estimator.base import mean_reduce
from jaqmc.utils.config import config_to_dict, configurable_dataclass, parse_config


class _BatchEstimator(Estimator):
    """Estimator that overrides evaluate_batch_walkers to skip vmap."""

    def __init__(self, key: str, value: float):
        self._key = key
        self._value = value

    def evaluate_batch_walkers(
        self, params, batched_data, prev_walker_stats, state, rngs
    ):
        del batched_data, prev_walker_stats
        return {self._key: jnp.array(self._value)}, state

    def reduce(self, walker_stats):
        return walker_stats

    def finalize_stats(self, batch_stats, state):
        mean_stats = jax.tree.map(lambda x: jnp.nanmean(x, axis=0), batch_stats)
        return {k: v * 2 for k, v in mean_stats.items()}


class _ToyData(Data):
    x: jnp.ndarray
    scale: jnp.ndarray


class _LocalValueEstimator(PerWalkerEstimator):
    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        del params, rngs
        return {"value": data.x * data.scale + prev_walker_stats["bias"]}, state


@dataclass
class _NestedEstimatorCollection(EstimatorCollection):
    first: Estimator
    second: Estimator


@dataclass
class _EstimatorCollection(EstimatorCollection):
    before: Estimator
    nested: _NestedEstimatorCollection
    after: Estimator


def test_estimator_collection_flattens_nested_estimators_in_order():
    before = _BatchEstimator("before", 1.0)
    first = _BatchEstimator("first", 2.0)
    second = _BatchEstimator("second", 3.0)
    after = _BatchEstimator("after", 4.0)

    flattened = dict(
        _EstimatorCollection(
            before=before,
            nested=_NestedEstimatorCollection(first=first, second=second),
            after=after,
        )
    )

    assert list(flattened) == ["before", "first", "second", "after"]
    assert flattened == {
        "before": before,
        "first": first,
        "second": second,
        "after": after,
    }


def test_estimator_collection_skips_estimators_disabled_after_construction():
    collection = _EstimatorCollection(
        before=_BatchEstimator("before", 1.0),
        nested=_NestedEstimatorCollection(
            first=_BatchEstimator("first", 2.0),
            second=_BatchEstimator("second", 3.0),
        ),
        after=_BatchEstimator("after", 4.0),
    )

    collection.nested.set_enabled(second=False)
    assert not collection.is_enabled("second")
    assert collection.is_enabled("first")
    assert list(collection) == ["before", "first", "after"]
    assert "second" not in collection

    collection.set_enabled(nested=False)
    assert not collection.is_enabled("first")
    assert list(collection) == ["before", "after"]
    assert list(EstimatorPipeline(collection).estimators) == ["before", "after"]

    with pytest.raises(AttributeError):
        collection.set_enabled(second=False)


def test_with_toggles_rejects_unknown_enabled_keys():
    @configurable_dataclass
    class _Collection(EstimatorCollection):
        first: _LocalValueEstimator = field(default_factory=_LocalValueEstimator)

    with pytest.raises(ValueError, match="do not match any field"):

        @configurable_dataclass
        class _BadSchema:
            estimators: Annotated[_Collection, WithToggles(typo=True)] = field(
                default_factory=_Collection
            )


def test_with_toggles_customizes_workflow_togglable_estimators():
    @configurable_dataclass
    class _SharedEstimators(EstimatorCollection):
        required: _LocalValueEstimator = field(default_factory=_LocalValueEstimator)
        optional: _LocalValueEstimator = field(default_factory=_LocalValueEstimator)

    @configurable_dataclass
    class _TrainSchema:
        estimators: Annotated[_SharedEstimators, WithToggles(optional=False)] = field(
            default_factory=_SharedEstimators
        )

    @configurable_dataclass
    class _EvalSchema:
        estimators: Annotated[
            _SharedEstimators, WithToggles(required=True, optional=False)
        ] = field(default_factory=_SharedEstimators)

    assert config_to_dict(_TrainSchema()) == {
        "estimators": {
            "required": {"vmap_chunk_size": None},
            "optional": {"enabled": False, "vmap_chunk_size": None},
        }
    }
    assert config_to_dict(_EvalSchema()) == {
        "estimators": {
            "required": {"enabled": True, "vmap_chunk_size": None},
            "optional": {"enabled": False, "vmap_chunk_size": None},
        }
    }

    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        parse_config(_TrainSchema, {}, ["estimators.required.enabled=false"])

    parsed_eval = parse_config(
        _EvalSchema,
        {},
        ["estimators.required.enabled=false", "estimators.optional.enabled=true"],
    )
    assert list(parsed_eval.estimators) == ["optional"]
    assert config_to_dict(parsed_eval)["estimators"]["required"]["enabled"] is False
    assert config_to_dict(parsed_eval)["estimators"]["optional"]["enabled"] is True


def test_pipeline_rejects_plain_estimator_functions():
    def evaluate(params, data, stats, state, rngs):
        return {"value": 1.0}, state

    with pytest.raises(TypeError, match="Estimator instances only"):
        EstimatorPipeline(cast(Any, {"value": evaluate}))


def test_local_estimator_chunks_default_vmap():
    batched_data = BatchedData(
        _ToyData(x=jnp.arange(7.0), scale=jnp.array(2.0)),
        fields_with_batch=["x"],
    )
    prev_walker_stats = {"bias": jnp.arange(7.0) * 10}
    params: dict[str, jnp.ndarray] = {}

    full_est = _LocalValueEstimator()
    chunked_est = _LocalValueEstimator(vmap_chunk_size=3)

    full, _ = full_est.evaluate_batch_walkers(
        params, batched_data, prev_walker_stats, None, jax.random.key(0)
    )
    chunked, _ = chunked_est.evaluate_batch_walkers(
        params, batched_data, prev_walker_stats, None, jax.random.key(0)
    )

    np.testing.assert_allclose(chunked["value"], full["value"])


def test_chunk_knob_only_lives_on_local_estimators():
    assert "vmap_chunk_size" not in Estimator.__dataclass_fields__
    assert "vmap_chunk_size" not in _BatchEstimator.__dataclass_fields__
    assert "vmap_chunk_size" in PerWalkerEstimator.__dataclass_fields__


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
    rngs = jax.random.key(0)
    step_stats, _ = pipeline.evaluate(None, None, state, rngs)
    assert set(step_stats.keys()) == {"x", "y"}
    assert step_stats["x"] == pytest.approx(1.0)
    assert step_stats["y"] == pytest.approx(2.0)


def test_pipeline_finalize_stats_averages_and_dispatches():
    pipeline, state = _make_pipeline_and_state()
    rngs = jax.random.key(0)
    step_stats, state = pipeline.evaluate(None, None, state, rngs)
    # Add batch dim of 1, same as VMC does
    batched = jax.tree.map(itemgetter(None), step_stats)
    final = pipeline.finalize_stats(batched, state)
    # _BatchEstimator.finalize_stats doubles values; averaging (1,) → scalar is identity
    assert final == {"x": 2.0, "y": 4.0}


def test_pipeline_finalize_stats_averages_stacked():
    pipeline, state = _make_pipeline_and_state()
    rngs = jax.random.key(0)
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


def test_mean_reduce_variance_stays_accurate_far_from_zero():
    values = jnp.asarray([-1000.0, -999.99], dtype=jnp.float32)
    got = mean_reduce({"energy": values})["energy_var"]
    ref = np.var(np.asarray(values, dtype=np.float64))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=0)
