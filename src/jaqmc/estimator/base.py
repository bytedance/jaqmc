# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass

type EstimateFn = Callable[
    [Params, Data, Mapping[str, Any], Any, PRNGKey],
    tuple[dict[str, Any], Any],
]
"""Signature for a plain function usable as an estimator's ``evaluate_local``."""

type EstimatorLike = Estimator | EstimateFn
"""Anything accepted where an :class:`Estimator` is expected."""


def mean_reduce(
    local_stats: Mapping[str, Any], include_variance: bool = True
) -> dict[str, Any]:
    """Reduce per-walker local values to step-level mean (and variance).

    Computes the mean over walkers and across devices.  When
    ``include_variance`` is True, appends ``{key}_var`` entries with
    the corresponding variance.

    Args:
        local_stats: Per-walker values (leading walker dimension).
        include_variance: Whether to include ``_var`` keys.

    Returns:
        Step-level statistics (walker dimension consumed).
    """
    stats = parallel_jax.pmean(
        jax.tree.map(lambda x: jnp.nanmean(x, axis=0), local_stats)
    )
    if include_variance:
        var_stats = jax.tree.map(
            lambda x, mean_x: parallel_jax.pmean(jnp.nanmean(x**2, axis=0)) - mean_x**2,
            local_stats,
            stats,
        )
        stats.update({f"{k}_var": v for k, v in var_stats.items()})
    return stats


@configurable_dataclass
class Estimator[DataT: Data]:
    """Base estimator with default no-op implementations.

    An estimator computes an observable quantity through a lifecycle
    with two output paths:

    **Stats path** (per-step statistics → final values):

    1. **evaluate** (``evaluate_local`` / ``evaluate_batch``) — compute
       per-walker local values.  For example, a kinetic energy estimator
       returns one energy scalar per walker.

    2. **reduce** — aggregate local values across walkers into per-step
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

    Subclass and override only the methods you need.

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

        Called once before the first ``evaluate`` call.

        Returns:
            State to thread through evaluate calls, or ``None``
            if no state is needed.
        """
        return None

    def evaluate_local(
        self,
        params: Params,
        data: DataT,
        prev_local_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        """Compute local values for a single walker.

        This is the main method to override.  The default
        ``evaluate_batch`` vmaps this over the walker dimension.

        Args:
            params: Wavefunction parameters.
            data: Data for a single walker.
            prev_local_stats: Local values produced by earlier
                estimators in the pipeline (single-walker).
            state: Estimator state from ``init`` or previous step.
            rngs: Random state.

        Returns:
            A tuple ``(local_stats, state)`` where ``local_stats``
            maps string keys to per-walker scalar or array values.
        """
        return {}, state

    def evaluate_batch(
        self,
        params: Params,
        batched_data: BatchedData[DataT],
        prev_local_stats: Mapping[str, Any],
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], Any]:
        """Compute local values over a batch of walkers.

        By default, vmaps ``evaluate_local`` over the walker dimension.
        Override directly for estimators that don't need per-walker
        vmapping (e.g. histogram aggregation, stats-only estimators).

        Args:
            params: Wavefunction parameters.
            batched_data: Batched sampled data.
            prev_local_stats: Local values produced by earlier
                estimators in the pipeline (with walker dimension).
            state: Estimator state.
            rngs: Random state.

        Returns:
            A tuple ``(local_stats, state)`` where ``local_stats``
            values have a leading walker dimension.
        """
        rngs = jax.random.split(rngs, batched_data.batch_size)
        return jax.vmap(
            self.evaluate_local,
            in_axes=(None, batched_data.vmap_axis, 0, None, 0),
        )(params, batched_data.data, prev_local_stats, state, rngs)

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Aggregate per-walker local values into per-step statistics.

        Called once per step after ``evaluate_batch``.  The output is
        what gets recorded by writers at each step.

        The default computes the mean (and variance) over walkers via
        ``mean_reduce``.

        Args:
            local_stats: This estimator's output from
                ``evaluate_batch`` (values have a walker dimension).

        Returns:
            Step-level statistics (walker dimension consumed).
        """
        return mean_reduce(local_stats)

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


class FunctionEstimator(Estimator):
    """Wraps a plain function as an :class:`Estimator`.

    The function is called as ``evaluate_local``; ``init``, ``reduce``,
    ``finalize_stats``, and ``finalize_state`` use the base-class defaults.
    """

    def __init__(self, fn: EstimateFn) -> None:
        self._fn = fn

    def evaluate_local(self, params, data, prev_local_stats, state, rngs):
        return self._fn(params, data, prev_local_stats, state, rngs)


class EstimatorPipeline:
    """Chains named estimators into an evaluate → reduce → finalize pipeline.

    Each estimator runs in insertion order. Later estimators can read
    earlier estimators' local values via ``prev_local_stats``.
    Key ownership is tracked so that :meth:`finalize_stats` dispatches each
    subset of statistics to the correct estimator.

    Args:
        estimators: Mapping from estimator name to either an
            :class:`Estimator` instance or a plain estimator function.
            Plain functions are wrapped in :class:`FunctionEstimator`.
    """

    def __init__(self, estimators: Mapping[str, EstimatorLike]) -> None:
        self.estimators = {
            name: est if isinstance(est, Estimator) else FunctionEstimator(est)
            for name, est in estimators.items()
        }
        self._reduce_keys: dict[str, frozenset[str]] = {}

    def __setitem__(self, name: str, est: EstimatorLike) -> None:
        self.estimators[name] = (
            est if isinstance(est, Estimator) else FunctionEstimator(est)
        )

    def __contains__(self, name: str) -> bool:
        return name in self.estimators

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
        """Compute averaged local values of the observables.

        Args:
            params: Wavefunction parameters.
            batched_data: Batched sampled data.
            state: Evaluator state.
            rngs: Random state.

        Returns:
            A tuple ``(step_stats, state)``, where ``step_stats`` is a flat
            dictionary merging each estimator's :meth:`~Estimator.reduce` output.
        """
        local_stats: dict[str, Any] = {}
        step_stats: dict[str, Any] = {}
        for name, estimator in self.estimators.items():
            rngs, sub_rngs = jax.random.split(rngs)
            stats_parts, state[name] = estimator.evaluate_batch(
                params, batched_data, local_stats, state[name], sub_rngs
            )
            local_stats.update(stats_parts)
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
