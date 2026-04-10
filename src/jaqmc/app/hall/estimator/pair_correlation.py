# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Pair correlation function estimator for the Haldane sphere.

Accumulates a histogram of geodesic pair angles :math:`\theta_{ij}`
weighted by :math:`1/\sin\theta_{ij}` to obtain the pair correlation
function :math:`g(\theta)`.

The normalization factor per evaluation step is included, but the
division by the total number of steps is **not** — divide the state
by the step count to obtain the final :math:`g(\theta)`.
"""

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class PairCorrelation(Estimator):
    r"""Pair correlation function :math:`g(\theta)` on the Haldane sphere.

    For each pair of electrons :math:`(i < j)`, computes the geodesic
    angle :math:`\theta_{ij}` and accumulates a histogram weighted by
    :math:`1/\sin\theta_{ij}`.

    Args:
        bins: Number of histogram bins.
    """

    bins: int = 200

    def init(self, data: Data, rngs: PRNGKey) -> jnp.ndarray:
        return jnp.zeros(self.bins)

    def evaluate_batch(
        self,
        params: Any,
        batched_data: BatchedData,
        prev_local_stats: Mapping[str, Any],
        state: jnp.ndarray,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        del params, prev_local_stats, rngs
        electrons = batched_data.data.electrons
        batch_size, nelec, _ = electrons.shape
        theta, phi = electrons[..., 0], electrons[..., 1]

        # Cartesian coordinates on the unit sphere
        sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
        xyz = jnp.stack(
            [sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t],
            axis=-1,
        )

        # Pairwise cosines and geodesic angles (upper triangle only)
        cos12 = jnp.sum(xyz[..., :, None, :] * xyz[..., None, :, :], axis=-1)
        pairs = cos12[:, *jnp.triu_indices(nelec, 1)]
        theta12 = jnp.arccos(jnp.clip(pairs, -1, 1)).reshape(-1)

        to_add, _ = jnp.histogram(
            theta12, self.bins, (0.0, jnp.pi), weights=1 / jnp.sin(theta12)
        )
        # Factor 2 converts (i < j) to (i != j); remaining factors normalize
        # to a density.  Division by number of evaluation steps is NOT included.
        return {}, state + to_add * 4 * self.bins / batch_size / nelec**2 / jnp.pi

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def finalize_stats(
        self, mean_stats: Mapping[str, Any], state: jnp.ndarray
    ) -> dict[str, Any]:
        return {}
