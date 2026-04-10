# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Base class for histogram-accumulating estimators with Kahan summation."""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator


class HistogramEstimator(Estimator):
    r"""Base for estimators that accumulate N-D histograms with Kahan summation.

    Subclasses must implement:

    - :meth:`_histogram_spec` — return ``(bins, ranges)`` for
      :func:`jnp.histogramdd`.
    - :meth:`extract` — return values to histogram, shape ``(..., ndim)``.

    The histogram and its Kahan compensation array are stored in the
    estimator state with a leading device dimension so that each device
    maintains a full independent histogram.  No per-step statistics are
    produced (:meth:`reduce` and :meth:`finalize_stats` return empty
    dicts).  The final histogram is produced by :meth:`finalize_state`,
    which sums across devices and returns raw counts.  The step count
    is provided by the work stage.  Normalization to physical density
    is left to post-processing since the correct measure depends on
    the coordinate system.

    **Kahan summation** prevents floating-point error from accumulating
    over tens of thousands of evaluation steps.  Each step's histogram
    counts are added via a compensated sum:

    .. math::

        y_n &= x_n - c_{n-1} \\
        t_n &= s_{n-1} + y_n \\
        c_n &= (t_n - s_{n-1}) - y_n \\
        s_n &= t_n

    where :math:`s` is the running sum, :math:`c` is the compensation
    term, and :math:`x_n` is the histogram counts at step *n*.

    Args:
        name: Key prefix for the output in the evaluation digest.
    """

    name: str = "density"

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        """Return ``(bins, ranges)`` for :func:`jnp.histogramdd`.

        ``bins`` is either a single int (same bin count for every
        dimension) or a tuple of per-dimension bin counts.  ``ranges``
        is a list of ``(min, max)`` tuples, one per histogram dimension.
        """
        raise NotImplementedError

    def extract(self, data: Data) -> jnp.ndarray:
        """Return values to histogram.

        The returned array is reshaped to ``(-1, ndim)`` before being
        passed to :func:`jnp.histogramdd`, where *ndim* is the number
        of ranges returned by :meth:`_histogram_spec`.
        """
        raise NotImplementedError

    def _histogram_shape(self) -> tuple[int, ...]:
        bins, ranges = self._histogram_spec()
        if isinstance(bins, int):
            return (bins,) * len(ranges)
        return tuple(bins)

    def init(self, data: Data, rngs: PRNGKey) -> dict[str, jnp.ndarray]:
        shape = self._histogram_shape()
        n = jax.device_count()
        return {
            "histogram": jnp.zeros((n, *shape)),
            "compensation": jnp.zeros((n, *shape)),
        }

    def evaluate_batch(
        self,
        params: Params,
        batched_data: BatchedData,
        prev_local_stats: Mapping[str, Any],
        state: dict[str, jnp.ndarray],
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        del params, prev_local_stats, rngs
        bins, ranges = self._histogram_spec()
        values = self.extract(batched_data.data)
        ndim = len(ranges)
        values = values.reshape(-1, ndim)
        counts = jnp.histogramdd(values, bins, ranges)[0]
        # Kahan summation (state arrays have a leading device dim that
        # broadcasts naturally with the per-device counts).
        adjusted = counts - state["compensation"]
        new_sum = state["histogram"] + adjusted
        new_comp = (new_sum - state["histogram"]) - adjusted
        return {}, {"histogram": new_sum, "compensation": new_comp}

    def reduce(self, local_stats: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def finalize_stats(
        self, mean_stats: Mapping[str, Any], state: dict[str, jnp.ndarray]
    ) -> dict[str, Any]:
        return {}

    def finalize_state(
        self, state: dict[str, jnp.ndarray], *, n_steps: int
    ) -> dict[str, Any]:
        """Sum histograms across devices.

        Returns:
            ``{self.name: histogram, self.name + ":n_steps": n_steps}``
            where *histogram* is the raw counts summed over all devices
            and *n_steps* is the total number of evaluation steps.
        """
        histogram = jnp.sum(state["histogram"], axis=0)
        return {self.name: histogram, f"{self.name}:n_steps": n_steps}
