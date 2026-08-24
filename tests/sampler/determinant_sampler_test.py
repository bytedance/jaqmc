# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.sampler.determinant import DeterminantMCMCSampler
from jaqmc.sampler.mcmc import gaussian_proposal


def test_proposal_changes_exactly_one_replica_per_walker():
    sampler = DeterminantMCMCSampler(n_states=3)
    electrons = jnp.zeros((16, 3, 2, 3))

    proposed = sampler.sampling_proposal(jax.random.key(7), electrons, 0.2)
    changed = jnp.any(proposed != electrons, axis=(2, 3))

    np.testing.assert_array_equal(changed.sum(axis=1), jnp.ones(16))


def test_m1_proposal_matches_native_physical_proposal():
    def deterministic_proposal(rngs, x, stddev):
        del rngs
        return x + stddev

    sampler = DeterminantMCMCSampler(
        n_states=1, sampling_proposal=deterministic_proposal
    )
    electrons = jnp.zeros((4, 1, 2, 3))
    proposed = sampler.sampling_proposal(jax.random.key(0), electrons, 0.5)
    np.testing.assert_allclose(proposed, 0.5)


def test_m1_random_proposal_has_exact_native_rng_semantics():
    sampler = DeterminantMCMCSampler(n_states=1)
    electrons = jnp.zeros((4, 1, 2, 3))
    key = jax.random.key(3)

    proposed = sampler.sampling_proposal(key, electrons, 0.5)
    expected = gaussian_proposal(key, electrons, 0.5)

    np.testing.assert_array_equal(proposed, expected)
