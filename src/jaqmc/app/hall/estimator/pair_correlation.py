# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Pair correlation function estimator for the Haldane sphere.

Accumulates a histogram of geodesic pair angles :math:`\theta_{ij}`
weighted by :math:`1/\sin\theta_{ij}`.

The histogram contains raw weighted counts.  To obtain the pair
correlation function :math:`g(\theta)`, multiply by
:math:`4\,b / (\pi\,N^2\,n_{\text{walkers}}\,n_{\text{steps}})`,
where :math:`b` is the number of bins, :math:`N` the number of
electrons, :math:`n_{\text{walkers}}` the global walker count
(``workflow.batch_size``), and :math:`n_{\text{steps}}` the
evaluation step count (provided as ``pair_correlation:n_steps``).
"""

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class PairCorrelation(HistogramEstimator):
    r"""Pair correlation function :math:`g(\theta)` on the Haldane sphere.

    For each pair of electrons :math:`(i < j)`, computes the geodesic
    angle :math:`\theta_{ij}` and accumulates a histogram weighted by
    :math:`1/\sin\theta_{ij}`.

    Args:
        bins: Number of histogram bins.
        data_field: Name of the coordinate field (runtime dep, default
            ``"electrons"``).
    """

    bins: int = 200
    data_field: str = runtime_dep(default="electrons")
    name: str = "pair_correlation"

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        return self.bins, [(0.0, jnp.pi)]

    def extract(self, data: Data) -> jnp.ndarray:
        """Return geodesic pair angles, shape ``(batch, n_pairs, 1)``."""
        electrons = data[self.data_field]
        nelec = electrons.shape[-2]
        theta, phi = electrons[..., 0], electrons[..., 1]

        sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
        xyz = jnp.stack(
            [sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t],
            axis=-1,
        )

        cos12 = jnp.sum(xyz[..., :, None, :] * xyz[..., None, :, :], axis=-1)
        pairs = cos12[..., *jnp.triu_indices(nelec, 1)]
        theta12 = jnp.arccos(jnp.clip(pairs, -1, 1))
        return theta12[..., None]

    def _weights(self, values: jnp.ndarray, data: Data) -> jnp.ndarray:
        r"""Return per-pair weights :math:`1/\sin\theta_{ij}`."""
        del data
        return 1.0 / jnp.sin(values[..., 0])
