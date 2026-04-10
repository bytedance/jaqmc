# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol

import jax
from jax import numpy as jnp

from jaqmc.array_types import ArrayLikeTree, Params, PRNGKey, PyTree
from jaqmc.data import BatchedData
from jaqmc.wavefunction import NumericWavefunctionEvaluate


class BatchLogProb(Protocol):
    def __call__(self, data: PyTree) -> jnp.ndarray:
        """Returns log-probability (``2 * log|psi|``) over a batch of walkers.

        Args:
            data: A batched PyTree (the data subset for this sampler's keys).

        Returns:
            A 1D array ``(batch_size,)`` with one log-probability per walker.
        """


class SamplerInit[StateT: ArrayLikeTree](Protocol):
    def __call__(self, data: PyTree, rngs: PRNGKey) -> StateT:
        """Returns initial sampler state.

        Args:
            data: Initial data.
            rngs: Random state.
        """


class SamplerStep[StateT: ArrayLikeTree](Protocol):
    def __call__(
        self,
        batch_log_prob: BatchLogProb,
        data: PyTree,
        state: StateT,
        rngs: PRNGKey,
    ) -> tuple[PyTree, dict[str, Any], StateT]:
        """Sample walker data.

        Args:
            batch_log_prob: Function to be sampled.
            data: Previously sampled data.
            state: State of the sampler.
            rngs: Random state.

        Returns:
            A tuple of
            - data: Sampled data.
            - stats: Statistical variables of the sampler.
            - state: New state of the sampler.

        Type Parameters:
            StateT: Sampler state type threaded through successive updates.
        """


class SamplerLike[StateT: ArrayLikeTree](Protocol):
    """Protocol for samplers.

    A sampler must have two callable attributes: ``init`` and ``step``.
    Defining them as methods on a class satisfies this protocol.

    Type Parameters:
        StateT: Sampler state type threaded through ``init`` and ``step``.
    """

    init: SamplerInit[StateT]
    step: SamplerStep[StateT]


class SamplePlan:
    """Coordinate how one or more samplers update batched walker data.

    In the common case, a workflow registers a single sampler for the
    ``"electrons"`` field. More complex systems can register different
    samplers for different fields, or update multiple fields together.

    A sample plan initializes sampler state, runs each sampler on its assigned
    part of the data, and combines the results back into one
    :class:`~jaqmc.data.BatchedData` object.

    Args:
        log_amplitude: Wavefunction log-amplitude used to score sampling
            proposals.
        samplers: Optional mapping from field names, or tuples of field names,
            to sampler instances.
    """

    def __init__(
        self,
        log_amplitude: NumericWavefunctionEvaluate,
        samplers: Mapping[str | tuple[str], SamplerLike] | None = None,
    ):
        self.log_amplitude = log_amplitude
        self.samplers: dict[tuple[str], SamplerLike] = {}
        if samplers is not None:
            for keys, sampler in samplers.items():
                self.will_sample(keys, sampler)

    def will_sample(self, keys: str | tuple[str], sampler: SamplerLike):
        """Register which data fields a sampler should update.

        Use this when one sampler should control a specific field such as
        ``"electrons"``, or when a sampler should update several fields
        together.

        Args:
            keys: Field name or field names handled by this sampler.
            sampler: Sampler instance that proposes updates for those fields.
        """
        if isinstance(keys, str):
            keys = (keys,)
        self.samplers[keys] = sampler

    def init(self, batched_data: BatchedData, rngs: PRNGKey) -> dict[str, Any]:
        """Initialize the state for all registered samplers.

        Each sampler receives the part of ``batched_data`` that it is
        responsible for and returns its own sampler state.

        Args:
            batched_data: Current batched walker data.
            rngs: Random key used for sampler initialization.

        Returns:
            A mapping of sampler states to pass back into :meth:`step`.
        """
        self.batch_log_amplitude = jax.vmap(
            self.log_amplitude, (None, batched_data.vmap_axis)
        )
        return {
            ",".join(keys): sampler.init(batched_data.data.subset(keys), rngs)
            for keys, sampler in self.samplers.items()
        }

    def step(
        self,
        params: Params,
        batched_data: BatchedData,
        state: dict[str, Any],
        rngs: PRNGKey,
    ) -> tuple[BatchedData, dict[str, Any], dict[str, Any]]:
        """Run one sampling round and return updated walker data.

        Each registered sampler proposes updates for its own fields, and the
        plan combines those updates into a new batched data object. Even when a
        sampler updates only part of the data, proposals are scored using the
        full wavefunction.

        Args:
            params: Wavefunction parameters.
            batched_data: Current batched walker data.
            state: Sampler state produced by :meth:`init` and previous calls to
                :meth:`step`.
            rngs: Random key for this sampling round.

        Returns:
            A tuple ``(batched_data, stats, state)`` containing the updated
            walker data, sampler statistics, and updated sampler state.
        """
        all_stats: dict[str, Any] = {}
        data = batched_data.data
        for keys, sampler in self.samplers.items():
            rngs, sub_rngs = jax.random.split(rngs)
            data_part, stats, state[",".join(keys)] = sampler.step(
                lambda x: 2 * self.batch_log_amplitude(params, data.merge(x)),
                batched_data.data.subset(keys),
                state[",".join(keys)],
                sub_rngs,
            )
            data = data.merge(data_part)
            all_stats.update(stats)
        return replace(batched_data, data=data), all_stats, state
