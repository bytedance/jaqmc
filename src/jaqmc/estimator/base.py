# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import reduce
from typing import Any, cast

import jax
from jax import numpy as jnp
from pydantic import (
    Field,
    GetCoreSchemaHandler,
    SerializerFunctionWrapHandler,
    TypeAdapter,
)
from pydantic_core import CoreSchema, core_schema

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.utils import parallel_jax
from jaqmc.utils.chunked_vmap import chunked_vmap
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class EstimatorCollection(Mapping):
    """Nested group of estimators that can be flattened for runtime use.

    The mapping view contains only estimators that are enabled (along with
    their enclosing groups). Workflows and parent collections configure which
    child fields expose a user-togglable ``enabled`` key via
    :func:`WithToggles`, and can toggle children programmatically via
    :meth:`set_enabled`.
    """

    def __post_init__(self) -> None:
        self._enabled: dict[str, bool] = {}
        self._key_map: dict[str, list[str]] = {}

        def visit(group: EstimatorCollection, prefix: Sequence[str] = ()) -> None:
            for f in fields(cast(Any, group)):
                value = getattr(group, f.name)
                if isinstance(value, EstimatorCollection):
                    visit(value, [*prefix, f.name])
                elif isinstance(value, Estimator):
                    if f.name in self._key_map:
                        raise ValueError(
                            f"Duplicate estimator name {f.name!r} in "
                            f"{type(self).__name__}."
                        )
                    self._key_map[f.name] = [*prefix, f.name]
                else:
                    raise TypeError(
                        f"{type(group).__name__}.{f.name} must be an "
                        f"Estimator or EstimatorCollection; got {type(value)}."
                    )

        visit(self)

    def set_enabled(self, **kwargs: bool) -> None:
        """Set the runtime enabled state for direct child estimators or subgroups.

        Raises:
            AttributeError: If any keyword does not match a field on this
                collection.
        """
        if unknown := set(kwargs) - {f.name for f in fields(cast(Any, self))}:
            raise AttributeError(
                f"{type(self).__name__} has no estimator field(s) {sorted(unknown)}."
            )
        self._enabled.update(kwargs)

    def is_enabled(self, key: str) -> bool:
        """Return whether the named estimator and its parent groups are enabled."""
        node: Any = self
        for name in self._key_map[key]:
            if not node._enabled.get(name, True):
                return False
            node = getattr(node, name)
        return True

    def __getitem__(self, key: str) -> "Estimator":
        if key not in self._key_map or not self.is_enabled(key):
            raise KeyError(key)
        return reduce(getattr, self._key_map[key], self)  # type: ignore

    def __iter__(self):
        return (key for key in self._key_map if self.is_enabled(key))

    def __len__(self):
        return sum(1 for _ in self)


@dataclass(frozen=True)
class _WithToggles:
    enabled: Mapping[str, bool]

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        known = {f.name for f in fields(source_type)}
        if unknown := set(self.enabled) - known:
            raise ValueError(
                f"{source_type.__name__} enabled keys {sorted(unknown)} "
                "do not match any field."
            )
        defaults = dict(self.enabled)

        def validate(raw: Any, nxt: Any) -> Any:
            overrides: dict[str, bool] = {}
            if isinstance(raw, dict):
                raw = {
                    k: dict(v) if k in defaults and isinstance(v, dict) else v
                    for k, v in raw.items()
                }
                if en_raw := {
                    k: raw[k].pop("enabled")
                    for k in defaults
                    if isinstance(raw.get(k), dict) and "enabled" in raw[k]
                }:
                    validator = TypeAdapter(dict[str, bool]).validator
                    overrides = validator.validate_python(en_raw)
            instance = nxt(raw)
            instance._enabled = {**defaults, **overrides}
            return instance

        def serialize(value: Any, nxt: SerializerFunctionWrapHandler) -> Any:
            d = nxt(value)
            for k, default in defaults.items():
                d[k] = {"enabled": value._enabled.get(k, default), **d[k]}
            return d

        return core_schema.no_info_wrap_validator_function(
            validate,
            handler(source_type),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                serialize, info_arg=False
            ),
        )


def WithToggles(**enabled: bool) -> Any:
    """Configure user-togglable estimators on an :class:`EstimatorCollection` field.

    Use with :data:`typing.Annotated` on an :class:`EstimatorCollection` field
    to declare which child estimators or subgroups expose an inline ``enabled``
    boolean configuration key and what their default state is.

    Args:
        **enabled: Mapping from child estimator or subgroup field name to its
            default ``enabled`` boolean state.

    Returns:
        A Pydantic field annotation attaching the toggle schema.
    """
    f = Field(validate_default=True)
    f.metadata.append(_WithToggles(enabled))
    return f


def mean_reduce(
    walker_stats: Mapping[str, Any], include_variance: bool = True
) -> dict[str, Any]:
    """Reduce per-walker values to step-level mean (and variance).

    Computes the mean over walkers and across devices.  When
    ``include_variance`` is True, appends ``{key}_var`` entries with
    the mean of squared deviations from that mean.  Deviations are
    centered before squaring so the result stays accurate in float32
    when the walker values lie far from zero.

    Args:
        walker_stats: Per-walker values (leading walker dimension).
        include_variance: Whether to include ``_var`` keys.

    Returns:
        Step-level statistics (walker dimension consumed).
    """
    stats = jax.tree.map(parallel_jax.pnanmean, walker_stats)
    if include_variance:
        var_stats = jax.tree.map(
            lambda x, mean_x: parallel_jax.pnanmean((x - mean_x) ** 2),
            walker_stats,
            stats,
        )
        stats.update({f"{k}_var": v for k, v in var_stats.items()})
    return stats


@configurable_dataclass
class Estimator[DataT: Data](ABC):
    """Base estimator with default no-op implementations.

    An estimator computes an observable quantity through a lifecycle
    with two output paths:

    **Stats path** (per-step statistics → final values):

    1. **evaluate** (``evaluate_single_walker`` / ``evaluate_batch_walkers``)
       — compute per-walker values. For example, a kinetic energy estimator
       returns one energy scalar per walker.

    2. **reduce** — aggregate per-walker values across walkers into per-step
       statistics.  The default computes mean and variance over walkers
       (via ``mean_reduce``).  The output is what gets written to disk
       at each step.

    3. **finalize_stats** — combine per-step statistics (with a leading
       step dimension) into final physical quantities.  This exists
       because some observables cannot be expressed as a single
       expectation — they require ratios, products, or other nonlinear
       combinations of step-level averages (e.g. overlap, polarization,
       energy gradients).

    **State path** (accumulated state → final values):

    4. **finalize_state** — extract final observables from accumulated
       estimator state.  Used by estimators that accumulate results
       directly in state (e.g. histograms) rather than through per-step
       statistics.  Called only during evaluation digest, never inside
       JIT.

    Subclass and override only the methods you need. Most estimators should
    inherit from :class:`PerWalkerEstimator`, which batches single-walker
    ``evaluate_single_walker`` implementations automatically. Inherit directly from
    this class when the estimator owns full-batch logic through
    ``evaluate_batch_walkers``.

    Runtime dependencies should be declared as
    :func:`~jaqmc.utils.wiring.runtime_dep` fields.
    They can be provided in two ways:

    1. **Programmatic**: pass directly in the constructor::

        estimator = EuclideanKinetic(f_log_psi=wf.evaluate)

    2. **Config-driven**: use ``wire(estimator, **context)``::

        wire(estimator, f_log_psi=wf.evaluate)

    Subclasses that need to compute derived state from runtime deps
    should do so in :meth:`init`.

    Type Parameters:
        DataT: Concrete one-walker ``Data`` subtype consumed by this estimator.
    """

    def init(self, data: DataT, rngs: PRNGKey) -> Any:
        """Initialize estimator state from an example data point.

        Called once before the first ``evaluate_batch_walkers`` call.

        Returns:
            State to thread through evaluation calls, or ``None``
            if no state is needed.
        """
        return None

    @abstractmethod
    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[DataT],
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        """Compute per-walker values over a batch of walkers.

        The base implementation is a no-op. Override this directly for
        estimators that own full-batch logic (e.g. histogram aggregation,
        stats-only estimators). Inherit from :class:`PerWalkerEstimator` for the
        common pattern where ``evaluate_single_walker`` is vmapped over walkers.

        Args:
            params: Wavefunction parameters.
            batched_data: Batched sampled data.
            prev_walker_stats: Per-walker values produced by earlier
                estimators in the pipeline (with walker dimension).
            state: Estimator state.
            rngs: Random state.

        Returns:
            A tuple ``(walker_stats, state)`` where ``walker_stats``
            values have a leading walker dimension.
        """

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Aggregate per-walker values into per-step statistics.

        Called once per step after ``evaluate_batch_walkers``. The output is
        what gets recorded by writers at each step.

        The default computes the mean (and variance) over walkers via
        ``mean_reduce``.

        Args:
            walker_stats: This estimator's output from
                ``evaluate_batch_walkers`` (values have a walker dimension).

        Returns:
            Step-level statistics (walker dimension consumed).
        """
        return mean_reduce(walker_stats)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: Any
    ) -> dict[str, Any]:
        """Combine per-step statistics into final physical quantities.

        Receives the ``reduce`` output accumulated over multiple steps,
        with a leading step dimension on every value.  Produces the
        final observable values.

        Override this when the observable requires nonlinear
        combinations of step-level averages (ratios, products, etc.).
        The default simply averages over steps.

        Args:
            batched_stats: This estimator's ``reduce`` output stacked
                over steps (values have a leading step dimension).
            state: Estimator state.

        Returns:
            Final observable values (step dimension consumed).
        """
        return jax.tree.map(lambda x: jnp.nanmean(x, axis=0), batched_stats)

    def finalize_state(self, state: Any, *, n_steps: int) -> dict[str, Any]:
        """Extract final observables from accumulated estimator state.

        Override this for estimators that accumulate results directly
        in state rather than through per-step statistics (e.g. histogram
        estimators).  Called only during evaluation digest, never inside
        JIT.

        Args:
            state: Estimator state after all evaluation steps.
            n_steps: Total number of evaluation steps completed.

        Returns:
            Final observable values derived from state.
        """
        return {}


@configurable_dataclass
class PerWalkerEstimator[DataT: Data](Estimator[DataT], ABC):
    """Estimator whose ``evaluate_single_walker`` is batched over walkers by vmap.

    Args:
        vmap_chunk_size: Number of walkers to evaluate per vmap chunk. Set to
            ``None`` to evaluate the full batch at once.
    """

    vmap_chunk_size: int | None = None

    @abstractmethod
    def evaluate_single_walker(
        self,
        params: Params,
        data: DataT,
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        """Compute per-walker values for a single walker.

        This is the main method to override when inheriting from
        :class:`PerWalkerEstimator`, which vmaps it over the walker dimension.

        Args:
            params: Wavefunction parameters.
            data: Data for a single walker.
            prev_walker_stats: Per-walker values produced by earlier
                estimators in the pipeline (single-walker).
            state: Estimator state from ``init`` or previous step.
            rngs: Random state.

        Returns:
            A tuple ``(walker_stats, state)`` where ``walker_stats``
            maps string keys to per-walker scalar or array values.
        """

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[DataT],
        prev_walker_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        """Vmap ``evaluate_single_walker`` over the walker dimension.

        Returns:
            A tuple ``(walker_stats, state)`` where ``walker_stats`` values have
            a leading walker dimension.
        """
        rngs = jax.random.split(rngs, batched_data.batch_size)
        return chunked_vmap(
            self.evaluate_single_walker,
            in_axes=(None, batched_data.vmap_axis, 0, None, 0),
            chunk_size=self.vmap_chunk_size,
        )(params, batched_data.data, prev_walker_stats, state, rngs)


class EstimatorPipeline:
    """Chains named estimators into an evaluate → reduce → finalize pipeline.

    Each estimator runs in insertion order. Later estimators can read
    earlier estimators' per-walker values via ``prev_walker_stats``.
    Key ownership is tracked so that :meth:`finalize_stats` dispatches each
    subset of statistics to the correct estimator.

    Args:
        estimators: Mapping from estimator name to :class:`Estimator` instances.
    """

    def __init__(
        self, estimators: EstimatorCollection | Mapping[str, Estimator]
    ) -> None:
        if not all(
            isinstance(estimator, Estimator) for estimator in estimators.values()
        ):
            raise TypeError("EstimatorPipeline accepts Estimator instances only.")
        self.estimators = dict(estimators.items())
        self._reduce_keys: dict[str, frozenset[str]] = {}

    def init(self, batched_data: BatchedData, rngs: PRNGKey) -> dict[str, Any]:
        """Initialize per-estimator state.

        Returns:
            ``{name: state}`` dict threaded through :meth:`evaluate`
            and :meth:`finalize_stats`.
        """
        data = batched_data.unbatched_example()
        return {
            name: estimator.init(data, sub_rngs)
            for (name, estimator), sub_rngs in zip(
                self.estimators.items(), jax.random.split(rngs, len(self.estimators))
            )
        }

    def evaluate(
        self,
        params: Params,
        batched_data: BatchedData,
        state: dict[str, Any],
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute averaged per-walker values of the observables.

        Args:
            params: Wavefunction parameters.
            batched_data: Batched sampled data.
            state: Evaluator state.
            rngs: Random state.

        Returns:
            A tuple ``(step_stats, state)``, where ``step_stats`` is a flat
            dictionary merging each estimator's :meth:`~Estimator.reduce` output.
        """
        walker_stats: dict[str, Any] = {}
        step_stats: dict[str, Any] = {}
        for name, estimator in self.estimators.items():
            rngs, sub_rngs = jax.random.split(rngs)
            stats_parts, state[name] = estimator.evaluate_batch_walkers(
                params, batched_data, walker_stats, state[name], sub_rngs
            )
            walker_stats.update(stats_parts)
            reduced = estimator.reduce(stats_parts)
            self._reduce_keys[name] = frozenset(reduced.keys())
            step_stats.update(reduced)

        return step_stats, state

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        """Finalize observables from per-step statistics.

        ``batched_stats`` must have a leading batch/step dimension on every
        value.  This method splits the result by key ownership and dispatches
        to each estimator's :meth:`~Estimator.finalize_stats`.

        In VMC, pass single-step stats with a batch dimension of 1
        (e.g. via ``tree.map(lambda x: x[None], step_stats)``).
        In evaluation, pass the stacked multi-step stats directly.

        Args:
            batched_stats: Flat statistics with a leading batch dimension.
            state: Evaluator state.

        Returns:
            Final values for the observables.

        Raises:
            RuntimeError: If :meth:`evaluate` was never called (key ownership
                is unknown).
        """
        if not self._reduce_keys:
            raise RuntimeError(
                "finalize_stats() called before evaluate(); key ownership is unknown."
            )
        final_stats: dict[str, Any] = {}
        for name, estimator in self.estimators.items():
            est_stats = {
                k: batched_stats[k]
                for k in self._reduce_keys[name]
                if k in batched_stats
            }
            final_stats.update(estimator.finalize_stats(est_stats, state[name]))
        return final_stats

    def digest(
        self,
        batched_stats: Mapping[str, Any],
        state: dict[str, Any],
        *,
        n_steps: int,
    ) -> dict[str, Any]:
        """Produce the full evaluation digest.

        Combines :meth:`finalize_stats` (from per-step statistics) with
        each estimator's :meth:`~Estimator.finalize_state` (from
        accumulated state).  Call this in evaluation digest, not inside
        JIT.

        Args:
            batched_stats: Flat statistics with a leading batch dimension.
            state: Evaluator state after all steps.
            n_steps: Total number of evaluation steps completed.

        Returns:
            Merged final values from both stats and state paths.
        """
        result = self.finalize_stats(batched_stats, state) if batched_stats else {}
        for name, estimator in self.estimators.items():
            result.update(estimator.finalize_state(state[name], n_steps=n_steps))
        return result
