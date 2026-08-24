# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Replica-row Metropolis sampler for determinant-state wavefunctions."""

from typing import Literal

import jax
from jax import numpy as jnp

from jaqmc.utils.config import configurable_dataclass

from .mcmc import MCMCSampler


@configurable_dataclass
class DeterminantMCMCSampler(MCMCSampler):
    """Reuse JaQMC MH adaptation while moving one physical replica per proposal.

    The initial implementation deliberately uses a full determinant recomputation
    for proposal scoring. It keeps the sampler contract identical to
    :class:`MCMCSampler`; a cached rank-one backend can be added without changing
    workflow or wavefunction interfaces.
    """

    n_states: int = 1
    update_mode: Literal["full"] = "full"

    def __post_init__(self):
        if self.n_states < 1:
            raise ValueError("n_states must be positive")
        if self.update_mode != "full":
            raise ValueError("Only the correctness-first 'full' update is supported")
        self._physical_proposal = self.sampling_proposal
        self.sampling_proposal = self._single_replica_proposal

    def _single_replica_proposal(self, rngs, x, stddev):
        if self.n_states == 1:
            return self._physical_proposal(rngs, x, stddev)
        leaves, treedef = jax.tree.flatten(x)
        if not leaves:
            return x
        batch_size = leaves[0].shape[0]
        rngs, index_key = jax.random.split(rngs)
        replica_index = jax.random.randint(
            index_key, (batch_size,), 0, self.n_states
        )
        selected = []
        for leaf in leaves:
            if leaf.ndim < 2 or leaf.shape[1] != self.n_states:
                raise ValueError(
                    "Determinant sampler fields must have shape [batch, states, ...]"
                )
            selected.append(
                jax.vmap(lambda row, index: row[index])(leaf, replica_index)
            )
        selected_tree = jax.tree.unflatten(treedef, selected)
        proposed_selected = self._physical_proposal(rngs, selected_tree, stddev)
        proposed_leaves = jax.tree.leaves(proposed_selected)
        proposed = []
        for leaf, proposal in zip(leaves, proposed_leaves):
            proposed.append(
                jax.vmap(lambda row, value, index: row.at[index].set(value))(
                    leaf, proposal, replica_index
                )
            )
        return jax.tree.unflatten(treedef, proposed)
