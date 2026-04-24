# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for density histogram estimators and Kahan summation."""

import math

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.data import BatchedData, Data
from jaqmc.estimator.density import (
    CartesianAxis,
    CartesianDensity,
    FractionalAxis,
    FractionalDensity,
    SphericalDensity,
)


class _TestData(Data):
    """Minimal data container with an electrons field."""

    electrons: jnp.ndarray


def _make_batched(electrons: jnp.ndarray) -> BatchedData:
    return BatchedData(
        data=_TestData(electrons=electrons),
        fields_with_batch=["electrons"],
    )


def _assert_bin(histogram: jnp.ndarray, index, expected: float) -> None:
    """Assert a histogram bin equals expected (exact integer counts)."""
    np.testing.assert_allclose(float(histogram[index]), expected)


KEY = jax.random.PRNGKey(0)


# -------------------------------------------------------------------
# Kahan summation
# -------------------------------------------------------------------


class TestKahanSummation:
    def test_kahan_preserves_precision(self):
        """Adding 1.0 to 2^24 in float32: naive loses it, Kahan keeps it."""
        large = jnp.float32(2**24)  # ULP = 2 at this scale
        small = jnp.float32(1.0)  # below ULP
        n_steps = 100

        naive = large
        for _ in range(n_steps):
            naive = naive + small

        kahan_sum, comp = large, jnp.float32(0.0)
        for _ in range(n_steps):
            adjusted = small - comp
            new_sum = kahan_sum + adjusted
            comp = (new_sum - kahan_sum) - adjusted
            kahan_sum = new_sum

        true_value = float(large) + n_steps
        # Naive lost all 100 additions
        np.testing.assert_allclose(float(naive), float(large))
        # Kahan preserved them
        assert abs(float(kahan_sum) - true_value) < 1.0

    def test_kahan_through_estimator(self):
        """Kahan in evaluate_batch preserves precision over many steps."""
        est = SphericalDensity(bins_theta=1)
        electrons = jnp.array([[[1.0, 0.0]]])
        batched = _make_batched(electrons)

        state: dict[str, jnp.ndarray] = {
            "histogram": jnp.array([[jnp.float32(2**24)]]),
            "compensation": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        for _ in range(100):
            _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)

        expected = float(2**24) + 100
        np.testing.assert_allclose(float(state["histogram"][0, 0]), expected, rtol=1e-6)


# -------------------------------------------------------------------
# SphericalDensity
# -------------------------------------------------------------------


class TestSphericalDensity:
    def test_1d_shape(self):
        est = SphericalDensity(bins_theta=20)
        data = _TestData(electrons=jnp.zeros((1, 2)))
        state = est.init(data, KEY)
        n = jax.device_count()
        assert state["histogram"].shape == (n, 20)
        assert state["compensation"].shape == (n, 20)

    def test_2d_shape(self):
        est = SphericalDensity(bins_theta=20, bins_phi=40)
        data = _TestData(electrons=jnp.zeros((1, 2)))
        state = est.init(data, KEY)
        n = jax.device_count()
        assert state["histogram"].shape == (n, 20, 40)

    def test_1d_binning(self):
        """Electrons at known theta values land in correct bins."""
        est = SphericalDensity(bins_theta=4)
        electrons = jnp.array([[[math.pi / 8, 0.0], [5 * math.pi / 8, 0.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 2)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        hist = state["histogram"][0]  # single-device histogram
        _assert_bin(hist, 0, 1.0)
        _assert_bin(hist, 2, 1.0)
        np.testing.assert_allclose(float(hist.sum()), 2.0)

    def test_2d_binning(self):
        """2-D histogram places electrons in correct 2D bins."""
        est = SphericalDensity(bins_theta=2, bins_phi=2)
        electrons = jnp.array([[[math.pi / 4, math.pi / 2]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 2)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], (0, 1), 1.0)

    def test_multi_step_accumulation(self):
        """Multiple evaluate_batch calls accumulate counts."""
        est = SphericalDensity(bins_theta=4)
        electrons = jnp.array([[[math.pi / 8, 0.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 2)))
        state = est.init(data, KEY)
        for _ in range(3):
            _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 0, 3.0)

    def test_reduce_returns_empty(self):
        assert SphericalDensity().reduce({}) == {}

    def test_finalize_stats_returns_empty(self):
        state = {
            "histogram": jnp.zeros(50),
            "compensation": jnp.zeros(50),
        }
        assert SphericalDensity().finalize_stats({}, state) == {}


# -------------------------------------------------------------------
# CartesianDensity
# -------------------------------------------------------------------


class TestCartesianDensity:
    def test_z_projection(self):
        """Projection onto z-axis extracts the z-coordinate."""
        est = CartesianDensity(
            axes={
                "z": CartesianAxis(direction=(0, 0, 1), bins=10, range=(0, 10)),
            }
        )
        electrons = jnp.array([[[0.0, 0.0, 5.5]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 5, 1.0)

    def test_direction_normalization(self):
        """Non-unit direction is normalized before projection."""
        est = CartesianDensity(
            axes={
                "d": CartesianAxis(direction=(3, 0, 0), bins=10, range=(0, 10)),
            }
        )
        electrons = jnp.array([[[5.0, 7.0, 3.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 5, 1.0)

    def test_oblique_direction(self):
        """Projection onto [1,1,0]: electron [3,3,0] -> 3*sqrt(2)."""
        est = CartesianDensity(
            axes={
                "diag": CartesianAxis(direction=(1, 1, 0), bins=10, range=(0, 5)),
            }
        )
        electrons = jnp.array([[[3.0, 3.0, 0.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 8, 1.0)

    def test_2d_histogram(self):
        """Two projection axes produce a 2D histogram."""
        est = CartesianDensity(
            axes={
                "x": CartesianAxis(direction=(1, 0, 0), bins=4, range=(0, 4)),
                "z": CartesianAxis(direction=(0, 0, 1), bins=4, range=(0, 4)),
            }
        )
        electrons = jnp.array([[[1.5, 0.0, 2.5]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        n = jax.device_count()
        assert state["histogram"].shape == (n, 4, 4)
        _assert_bin(state["histogram"][0], (1, 2), 1.0)

    def test_none_disables_axis(self):
        """Setting a projection to None removes it from the histogram."""
        est = CartesianDensity(
            axes={
                "x": CartesianAxis(direction=(1, 0, 0), bins=4, range=(0, 4)),
                "y": None,
                "z": CartesianAxis(direction=(0, 0, 1), bins=4, range=(0, 4)),
            }
        )
        electrons = jnp.array([[[1.5, 0.0, 2.5]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        # Only x and z remain -> 2D histogram
        n = jax.device_count()
        assert state["histogram"].shape == (n, 4, 4)
        _assert_bin(state["histogram"][0], (1, 2), 1.0)

    def test_none_reduces_to_1d(self):
        """Disabling all but one axis produces a 1D histogram."""
        est = CartesianDensity(
            axes={
                "x": None,
                "y": None,
                "z": CartesianAxis(direction=(0, 0, 1), bins=10, range=(0, 10)),
            }
        )
        electrons = jnp.array([[[0.0, 0.0, 5.5]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        n = jax.device_count()
        assert state["histogram"].shape == (n, 10)
        _assert_bin(state["histogram"][0], 5, 1.0)


# -------------------------------------------------------------------
# FractionalDensity
# -------------------------------------------------------------------


class TestFractionalDensity:
    def test_cubic_cell(self):
        """Cubic cell: fractional coord = position / cell_length."""
        cell_length = 10.0
        inv_lattice = jnp.linalg.inv(cell_length * jnp.eye(3))
        est = FractionalDensity(
            axes={"c": FractionalAxis(lattice_index=2, bins=10)},
            inv_lattice=inv_lattice,
        )
        electrons = jnp.array([[[0.0, 0.0, 5.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 5, 1.0)

    def test_non_orthogonal_cell(self):
        """Non-orthogonal lattice: fractional coords are correct."""
        lattice = jnp.array(
            [
                [10.0, 0.0, 0.0],
                [5.0, 8.66, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        inv_lattice = jnp.linalg.inv(lattice)
        est = FractionalDensity(
            axes={"a": FractionalAxis(lattice_index=0, bins=10)},
            inv_lattice=inv_lattice,
        )
        # [5,0,0] with a=[10,0,0] gives frac_a = 0.5 -> bin 5
        electrons = jnp.array([[[5.0, 0.0, 0.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 5, 1.0)

    def test_wrapping(self):
        """Positions outside [0,1) wrap via modulo."""
        inv_lattice = jnp.linalg.inv(10.0 * jnp.eye(3))
        est = FractionalDensity(
            axes={"c": FractionalAxis(lattice_index=2, bins=10)},
            inv_lattice=inv_lattice,
        )
        # z=15 -> frac=1.5 -> wraps to 0.5 -> bin 5
        electrons = jnp.array([[[0.0, 0.0, 15.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        _assert_bin(state["histogram"][0], 5, 1.0)

    def test_2d_axes(self):
        """Two fractional axes produce a 2D histogram."""
        inv_lattice = jnp.linalg.inv(10.0 * jnp.eye(3))
        est = FractionalDensity(
            axes={
                "a": FractionalAxis(lattice_index=0, bins=5),
                "b": FractionalAxis(lattice_index=1, bins=5),
            },
            inv_lattice=inv_lattice,
        )
        # frac = (0.3, 0.7) -> bins (1, 3) for 5-bin [0,1)
        electrons = jnp.array([[[3.0, 7.0, 0.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        n = jax.device_count()
        assert state["histogram"].shape == (n, 5, 5)
        _assert_bin(state["histogram"][0], (1, 3), 1.0)

    def test_none_disables_axis(self):
        """Setting an axis to None removes it from the histogram."""
        inv_lattice = jnp.linalg.inv(10.0 * jnp.eye(3))
        est = FractionalDensity(
            axes={
                "a": None,
                "b": None,
                "c": FractionalAxis(lattice_index=2, bins=10),
            },
            inv_lattice=inv_lattice,
        )
        electrons = jnp.array([[[0.0, 0.0, 5.0]]])
        batched = _make_batched(electrons)
        data = _TestData(electrons=jnp.zeros((1, 3)))
        state = est.init(data, KEY)
        _, state = est.evaluate_batch_walkers(None, batched, {}, state, KEY)
        n = jax.device_count()
        assert state["histogram"].shape == (n, 10)
        _assert_bin(state["histogram"][0], 5, 1.0)
