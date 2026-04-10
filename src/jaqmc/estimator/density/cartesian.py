# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Cartesian density estimator for molecules and open-boundary systems."""

from dataclasses import field

from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.histogram import HistogramEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class CartesianAxis:
    """Configuration for one histogram axis in Cartesian coordinates.

    Args:
        direction: Direction vector to project positions onto.
            Normalized internally — need not be a unit vector.
        bins: Number of histogram bins.
        range: ``(min, max)`` bounds for the projected coordinate.
    """

    direction: tuple[float, ...] = (0.0, 0.0, 1.0)
    bins: int = 50
    range: tuple[float, float] = (0.0, 1.0)


@configurable_dataclass
class CartesianDensity(HistogramEstimator):
    """Electron density along arbitrary directions in Cartesian space.

    Each histogram axis is defined by a direction, bin count, and range.
    Direction vectors are normalized internally, so they need not be
    unit vectors.

    For periodic systems where lattice-aligned density is needed, use
    :class:`~jaqmc.estimator.density.FractionalDensity` instead.

    Args:
        axes: Per-axis configuration keyed by user-chosen labels.
            Set a value to ``None`` to disable an axis inherited from
            defaults (e.g. when the workflow provides x/y/z but you
            only want z).
    """

    axes: dict[str, CartesianAxis | None] = field(default_factory=dict)

    def _sorted_axes(self) -> list[CartesianAxis]:
        return [a for a in self.axes.values() if a is not None]

    def _histogram_spec(
        self,
    ) -> tuple[int | tuple[int, ...], list[tuple[float, float]]]:
        axes = self._sorted_axes()
        bins = tuple(a.bins for a in axes)
        ranges = [a.range for a in axes]
        return bins, ranges

    def extract(self, data: Data) -> jnp.ndarray:
        axes = self._sorted_axes()
        dirs = jnp.array([a.direction for a in axes])
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        return data["electrons"] @ dirs.T
