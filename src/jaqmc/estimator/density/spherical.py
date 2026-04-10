# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Spherical density estimator for the Haldane sphere."""

import math

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class SphericalDensity(HistogramEstimator):
    r"""Electron density on the Haldane sphere.

    Accumulates a histogram of electron positions in spherical
    coordinates.  By default only the polar angle
    :math:`\theta \in [0, \pi]` is binned (1-D histogram).  Setting
    ``bins_phi`` enables a 2-D :math:`(\theta, \varphi)` histogram
    with :math:`\varphi \in [-\pi, \pi]`.

    Args:
        bins_theta: Number of bins for the polar angle.
        bins_phi: Number of bins for the azimuthal angle.
            ``None`` (default) produces a 1-D theta-only histogram.
    """

    bins_theta: int = 50
    bins_phi: int | None = None

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        if self.bins_phi is None:
            return self.bins_theta, [(0.0, math.pi)]
        return (self.bins_theta, self.bins_phi), [(0.0, math.pi), (-math.pi, math.pi)]

    def extract(self, data: Data) -> jnp.ndarray:
        ndim = 1 if self.bins_phi is None else 2
        return data["electrons"][..., :ndim]
