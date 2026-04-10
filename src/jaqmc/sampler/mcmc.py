# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple, Protocol

import jax
from jax import lax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey, PyTree
from jaqmc.utils import parallel_jax
from jaqmc.utils.array import match_first_axis_of
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

from .base import BatchLogProb, SamplerLike

__all__ = ["MCMCSampler"]  # Keep it as the "default export"
__all__ += ["MCMCState", "SamplingProposal"]


class SamplingProposal[StateT: PyTree](Protocol):
    def __call__(self, rngs: PRNGKey, x: StateT, stddev: float | jnp.ndarray) -> StateT:
        """Propose a new sample from the current state.

        Args:
            rngs: Random state.
            x: Current sample/state.
            stddev: Proposal width parameter.

        Returns:
            Proposed sample/state with the same structure as ``x``.

        Type Parameters:
            StateT: PyTree sample/state type preserved by the proposal.
        """


class MCMCState(NamedTuple):
    """State carried by the adaptive Metropolis-Hastings sampler.

    Attributes:
        stddev: Current proposal width.
        pmoves: Rolling acceptance-rate history used for width adaptation.
        counter: Number of completed sampler steps.
    """

    stddev: jnp.ndarray
    pmoves: jnp.ndarray
    counter: jnp.ndarray


def gaussian_proposal(rngs: PRNGKey, x, stddev: float | jnp.ndarray):
    return jax.tree.map(lambda a: a + jax.random.normal(rngs, a.shape) * stddev, x)


@configurable_dataclass
class MCMCSampler(SamplerLike[MCMCState]):
    """Metropolis-Hastings MCMC sampler.

    Args:
        steps: Number of Metropolis-Hastings updates per sample draw.
            Controls decorrelation between consecutive samples. Also
            determines the granularity of burn-in (see
            ``WorkStageConfig.burn_in``).
        initial_width: Initial width (stddev) of the Gaussian proposal.
        adapt_frequency: Frequency of adaptive width updates.
        pmove_range: Target range for acceptance rate.
        sampling_proposal: Proposal function for MCMC moves.
    """

    steps: int = 10
    initial_width: float = 0.1
    adapt_frequency: int = 100
    pmove_range: tuple[float, float] = (0.5, 0.55)
    sampling_proposal: SamplingProposal = runtime_dep(default=gaussian_proposal)

    def init(self, data, rngs):
        """Initialize adaptive Metropolis-Hastings sampler state.

        Args:
            data: Current sample data. Not used.
            rngs: Random key. Not used.

        Returns:
            Initial :class:`MCMCState` containing proposal width, acceptance
            history, and adaptation counter.
        """
        del data, rngs
        return MCMCState(
            stddev=jnp.array(self.initial_width),
            pmoves=jnp.zeros(self.adapt_frequency),
            counter=jnp.array(0),
        )

    def _mh_update[StateT: PyTree](
        self,
        batch_log_prob: BatchLogProb,
        x1: StateT,
        rngs: PRNGKey,
        log_prob_1: jnp.ndarray,
        num_accepts: jnp.ndarray,
        stddev: float | jnp.ndarray = 0.02,
    ) -> tuple[StateT, PRNGKey, jnp.ndarray, jnp.ndarray]:
        """Performs one Metropolis-Hastings step using an all-electron move.

        Args:
            batch_log_prob: Function to sample returning the log probability.
            x1: Initial MCMC configurations.
            rngs: Random state.
            log_prob_1: Log probability of f evaluated at x1.
            num_accepts: Number of MH move proposals accepted.
            stddev: Width of Gaussian move proposal.

        Returns:
            (x, rngs, log_prob, num_accepts), where:
            - x: Updated MCMC configurations.
            - rngs: Random state.
            - log_prob: Log probability of wf evaluated at x.
            - num_accepts: Update running total of number of accepted MH moves.

        Type Parameters:
            StateT: PyTree state/data type for the proposed and accepted samples.
        """
        rng_new, rng_sample, rng_cond = jax.random.split(rngs, 3)
        x2 = self.sampling_proposal(rng_sample, x1, stddev)
        log_prob_2 = batch_log_prob(x2).real
        ratio = log_prob_2 - log_prob_1

        rnd = jnp.log(jax.random.uniform(rng_cond, shape=log_prob_1.shape))
        cond = ratio > rnd
        x_new = jax.tree.map(
            lambda x1, x2: jnp.where(match_first_axis_of(cond, x1), x2, x1), x1, x2
        )
        log_prob_new = jnp.where(cond, log_prob_2, log_prob_1)
        num_accepts += jnp.sum(cond)
        return x_new, rng_new, log_prob_new, num_accepts

    def step[StateT: PyTree](
        self,
        batch_log_prob: BatchLogProb,
        data: StateT,
        state: MCMCState,
        rngs: PRNGKey,
    ) -> tuple[StateT, dict[str, Any], MCMCState]:
        """Run multiple MH updates and adapt proposal width.

        Args:
            batch_log_prob: Log-probability function over a batch.
            data: Current MCMC configurations.
            state: Sampler state.
            rngs: Random state.

        Returns:
            Tuple of ``(data, stats, new_state)`` after one sampler step.

        Type Parameters:
            StateT: PyTree sample/state type threaded through the step.

        Raises:
            ValueError: If ``batch_log_prob(data)`` does not return shape
                ``(batch_size,)``.
        """
        logprob = batch_log_prob(data).real
        if logprob.ndim != 1:
            raise ValueError(
                f"log_amplitude should return a scalar, got shape {logprob.shape[1:]}."
            )
        num_accepts = parallel_jax.pvary(jnp.array(0.0))
        data, _, _, num_accepts = lax.fori_loop(
            0,
            self.steps,
            lambda _, x: self._mh_update(batch_log_prob, *x, stddev=state.stddev),  # type: ignore
            (data, rngs, logprob, num_accepts),
        )
        pmove = jnp.sum(num_accepts) / (self.steps * logprob.shape[0])
        pmove = parallel_jax.pmean(pmove)

        # Adaptive MCMC move width
        stddev, pmoves, counter = state
        counter += 1
        t_since_mcmc_update = counter % self.adapt_frequency
        pmoves = pmoves.at[t_since_mcmc_update].set(pmove)
        stddev = jnp.where(
            t_since_mcmc_update == 0,
            jnp.where(
                jnp.mean(pmoves) > self.pmove_range[1],
                stddev * 1.1,
                jnp.where(jnp.mean(pmoves) < self.pmove_range[0], stddev / 1.1, stddev),
            ),
            stddev,
        )
        new_state = MCMCState(counter=counter, pmoves=pmoves, stddev=stddev)
        return data, {"pmove": pmove}, new_state
