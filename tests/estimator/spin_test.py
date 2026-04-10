# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the S^2 estimator using mock wavefunctions with known values.

For a constant wavefunction (psi = const for all configurations), every swap
ratio psi(swap)/psi = 1. This gives:

    S^2 = S_z(S_z + 1) + n_minority - n_minority * n_majority
        = S_z(S_z + 1) + n_min * (1 - n_maj)

This is analytically checkable without running any real physics.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator.spin import SpinSquared
from jaqmc.utils.wiring import wire


class _ElectronData(Data):
    """Minimal Data subclass with just an electrons field."""

    electrons: jnp.ndarray


def _make_spin_estimator(n_up, n_down, ndim=3):
    """Create a SpinSquared estimator with a constant mock wavefunction.

    Returns:
        Tuple of (estimator, data, rng_key).
    """
    est = SpinSquared()

    # Constant wavefunction: phase=1.0 (real positive), log|psi|=0.0
    def phase_logpsi(params, data):
        return jnp.array(1.0), jnp.array(0.0)

    wire(est, n_up=n_up, n_down=n_down, phase_logpsi=phase_logpsi)

    n_elec = n_up + n_down
    data = _ElectronData(electrons=jnp.ones((n_elec, ndim)))
    key = jax.random.key(0)
    est.init(data, key)
    return est, data, key


def _expected_s2_constant_wf(n_up, n_down):
    """Analytical S^2 for a constant wavefunction where all swap ratios = 1.

    S^2 = S_z(S_z+1) + n_min - n_min * n_maj
        = S_z(S_z+1) + n_min * (1 - n_maj)

    Returns:
        Expected S^2 value.
    """
    sz = abs(n_up - n_down) * 0.5
    n_min = min(n_up, n_down)
    n_maj = max(n_up, n_down)
    return sz * (sz + 1) + n_min * (1 - n_maj)


@pytest.mark.parametrize(
    "n_up,n_down",
    [
        pytest.param(1, 1, id="1up-1down"),
        pytest.param(2, 1, id="2up-1down"),
        pytest.param(2, 2, id="2up-2down"),
        pytest.param(3, 1, id="3up-1down"),
        pytest.param(3, 2, id="3up-2down"),
        pytest.param(3, 3, id="3up-3down"),
    ],
)
def test_s2_constant_wavefunction(n_up, n_down):
    """S^2 with constant wavefunction matches analytical formula."""
    est, data, key = _make_spin_estimator(n_up, n_down)
    stats, _ = est.evaluate_local(
        params={}, data=data, prev_local_stats={}, state=None, rngs=key
    )
    expected = _expected_s2_constant_wf(n_up, n_down)
    np.testing.assert_allclose(float(stats["spin:s2"]), expected, atol=1e-5)


def test_s2_fully_polarized():
    """Fully polarized state (n_down=0): S^2 = S_z(S_z+1), no swap terms."""
    n_up, n_down = 3, 0
    est = SpinSquared()

    # Wavefunction should never be called for swaps (no minority electrons)
    def phase_logpsi(params, data):
        return jnp.array(1.0), jnp.array(0.0)

    wire(est, n_up=n_up, n_down=n_down, phase_logpsi=phase_logpsi)

    data = _ElectronData(electrons=jnp.ones((n_up, 3)))
    key = jax.random.key(0)
    est.init(data, key)
    stats, _ = est.evaluate_local(
        params={}, data=data, prev_local_stats={}, state=None, rngs=key
    )
    expected = 1.5 * 2.5  # S_z=3/2, S^2=S_z(S_z+1)
    np.testing.assert_allclose(float(stats["spin:s2"]), expected, atol=1e-5)


def test_s2_singlet_identical_orbitals():
    """Two electrons, constant wavefunction: swap ratio = 1, S^2 = 0.

    For a constant wavefunction with n_up=1, n_down=1:
    S^2 = 0*(0+1) + 1 - 1*1 = 0 (singlet).
    """
    est, data, key = _make_spin_estimator(1, 1)
    stats, _ = est.evaluate_local(
        params={}, data=data, prev_local_stats={}, state=None, rngs=key
    )
    np.testing.assert_allclose(float(stats["spin:s2"]), 0.0, atol=1e-5)


def test_s2_swap_indices_correct():
    """Verify the minority/majority index assignment."""
    est_balanced = SpinSquared()
    wire(
        est_balanced,
        n_up=2,
        n_down=2,
        phase_logpsi=lambda p, d: (jnp.array(1.0), jnp.array(0.0)),
    )
    data = _ElectronData(electrons=jnp.ones((4, 3)))
    est_balanced.init(data, jax.random.key(0))

    # When n_up == n_down, minority = up (indices 0..n_up-1)
    np.testing.assert_array_equal(est_balanced._minority_idx, jnp.array([0, 1]))
    np.testing.assert_array_equal(est_balanced._majority_idx, jnp.array([2, 3]))

    est_unbalanced = SpinSquared()
    wire(
        est_unbalanced,
        n_up=3,
        n_down=1,
        phase_logpsi=lambda p, d: (jnp.array(1.0), jnp.array(0.0)),
    )
    data = _ElectronData(electrons=jnp.ones((4, 3)))
    est_unbalanced.init(data, jax.random.key(0))

    # When n_up > n_down, minority = down (indices n_up..n_elec-1)
    np.testing.assert_array_equal(est_unbalanced._minority_idx, jnp.array([3]))
    np.testing.assert_array_equal(est_unbalanced._majority_idx, jnp.array([0, 1, 2]))
