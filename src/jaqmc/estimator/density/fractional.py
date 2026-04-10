# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fractional-coordinate density estimator for periodic systems."""

from dataclasses import field

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class FractionalAxis:
    """Configuration for one histogram axis in fractional coordinates.

    Args:
        lattice_index: Which fractional coordinate to histogram
            (0, 1, or 2 for the first, second, or third lattice vector).
        bins: Number of histogram bins.
    """

    lattice_index: int = 0
    bins: int = 50


@configurable_dataclass
class FractionalDensity(HistogramEstimator):
    r"""Electron density in fractional (lattice) coordinates.

    Converts Cartesian electron positions to fractional coordinates
    via the inverse lattice matrix:

    .. math::

        \mathbf{f} = L^{-1}\,\mathbf{r} \mod 1

    then histograms selected fractional axes.  The range is always
    :math:`[0, 1)` per axis regardless of cell shape.

    For molecules or other open-boundary systems, use
    :class:`~jaqmc.estimator.density.CartesianDensity` instead.

    Args:
        axes: Per-axis configuration keyed by user-chosen labels.
            Set a value to ``None`` to disable an axis inherited from
            defaults.
        inv_lattice: Inverse lattice matrix, shape ``(3, 3)``.
            Set by the workflow via :func:`~jaqmc.utils.wiring.wire`.
    """

    axes: dict[str, FractionalAxis | None] = field(default_factory=dict)
    inv_lattice: jnp.ndarray = runtime_dep()

    def _sorted_axes(self) -> list[FractionalAxis]:
        return sorted(
            (a for a in self.axes.values() if a is not None),
            key=lambda a: a.lattice_index,
        )

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        axes = self._sorted_axes()
        bins = tuple(a.bins for a in axes)
        ranges = [(0.0, 1.0)] * len(axes)
        return bins, ranges

    def extract(self, data: Data) -> jnp.ndarray:
        frac = data["electrons"] @ self.inv_lattice.T % 1.0
        indices = jnp.array([a.lattice_index for a in self._sorted_axes()])
        return frac[..., indices]
