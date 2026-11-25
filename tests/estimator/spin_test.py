# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SpinSquared estimator and penalty gradients."""

from operator import itemgetter
from typing import Any

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.estimator.loss_grad import LossGrad
from jaqmc.estimator.spin import SpinSquared
from jaqmc.utils.wiring import wire


def _mock_all_gather_identity(monkeypatch):
    monkeypatch.setattr(
        lax,
        "all_gather",
        lambda x, axis_name, tiled=True: x,
    )


class _ElectronData(Data):
    """Minimal Data subclass with just an electrons field."""

    electrons: jnp.ndarray


class _PositionData(Data):
    """Minimal Data subclass with a custom positions field."""

    positions: jnp.ndarray


def _make_spin_estimator(n_up, n_down, ndim=3, mode="full"):
    """Create a spin estimator with a constant mock wavefunction.

    Returns:
        Tuple of (estimator, data, state, rng_key).
    """
    est = SpinSquared(mode=mode)

    # Constant wavefunction: phase=1.0 (real positive), log|psi|=0.0
    def phase_logpsi(
        params: Any, data: Data, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del params, data, precision
        return jnp.array(1.0), jnp.array(0.0)

    wire(est, n_up=n_up, n_down=n_down, phase_logpsi=phase_logpsi)

    n_elec = n_up + n_down
    data = _ElectronData(electrons=jnp.ones((n_elec, ndim)))
    key = jax.random.key(0)
    state = est.init(data, key)
    return est, data, state, key


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
@pytest.mark.parametrize("mode", ["full", "single_electron"])
def test_spin_constant_wavefunction(n_up, n_down, mode):
    """SpinSquared modes with constant wavefunction match analytical formula."""
    est, data, state, key = _make_spin_estimator(n_up, n_down, mode=mode)
    stats, _ = est.evaluate_single_walker(
        params={}, data=data, prev_walker_stats={}, state=state, rngs=key
    )
    expected = _expected_s2_constant_wf(n_up, n_down)
    np.testing.assert_allclose(float(stats["spin_squared"]), expected, atol=1e-5)


@pytest.mark.parametrize("mode", ["full", "single_electron"])
def test_spin_fully_polarized(monkeypatch, mode):
    """Fully polarized state (n_down=0): S^2 = S_z(S_z+1) and zero penalty gradient."""
    _mock_all_gather_identity(monkeypatch)
    n_up, n_down = 3, 0
    est = SpinSquared(mode=mode, loss_weight=2.0)

    # Wavefunction should never be called for swaps (no minority electrons)
    def phase_logpsi(
        params: Any, data: Data, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del params, data, precision
        return jnp.array(1.0), jnp.array(0.0)

    wire(
        est,
        n_up=n_up,
        n_down=n_down,
        phase_logpsi=phase_logpsi,
    )

    pipeline = EstimatorPipeline(
        {
            "spin": est,
            "grads": LossGrad(
                f_log_psi=lambda p, d, *, precision=None: phase_logpsi(
                    p, d, precision=precision
                )[1]
            ),
        }
    )
    batched_data = BatchedData(
        _ElectronData(electrons=jnp.ones((2, n_up, 3))), ["electrons"]
    )
    key = jax.random.key(0)
    state = pipeline.init(batched_data, key)
    params = {"w": jnp.array([1.5, -0.5])}
    step_stats, state = pipeline.evaluate(params, batched_data, state, key)
    final = pipeline.finalize_stats(jax.tree.map(itemgetter(None), step_stats), state)
    expected = 1.5 * 2.5  # S_z=3/2, S^2=S_z(S_z+1)
    np.testing.assert_allclose(float(final["spin_squared"]), expected, atol=1e-5)
    np.testing.assert_allclose(float(final["loss:spin_squared"]), 0.0)
    np.testing.assert_allclose(final["grads"]["w"], jnp.zeros_like(params["w"]))


def test_s2_singlet_identical_orbitals():
    """Two electrons, constant wavefunction: swap ratio = 1, S^2 = 0.

    For a constant wavefunction with n_up=1, n_down=1:
    S^2 = 0*(0+1) + 1 - 1*1 = 0 (singlet).
    """
    est, data, state, key = _make_spin_estimator(1, 1)
    stats, _ = est.evaluate_single_walker(
        params={}, data=data, prev_walker_stats={}, state=state, rngs=key
    )
    np.testing.assert_allclose(float(stats["spin_squared"]), 0.0, atol=1e-5)


def test_s2_swap_indices_correct():
    """Verify the minority/majority index assignment."""
    est_balanced = SpinSquared()
    wire(
        est_balanced,
        n_up=2,
        n_down=2,
        phase_logpsi=lambda p, d, *, precision=None: (jnp.array(1.0), jnp.array(0.0)),
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
        phase_logpsi=lambda p, d, *, precision=None: (jnp.array(1.0), jnp.array(0.0)),
    )
    data = _ElectronData(electrons=jnp.ones((4, 3)))
    est_unbalanced.init(data, jax.random.key(0))

    # When n_up > n_down, minority = down (indices n_up..n_elec-1)
    np.testing.assert_array_equal(est_unbalanced._minority_idx, jnp.array([3]))
    np.testing.assert_array_equal(est_unbalanced._majority_idx, jnp.array([0, 1, 2]))


def test_s2_custom_data_field():
    """SpinSquared should swap coordinates from the configured data field."""
    est = SpinSquared(data_field="positions")

    def phase_logpsi(
        params: Any, data: _PositionData, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del params, precision
        positions = data["positions"]
        return jnp.array(1.0), positions[0, 0] - positions[1, 0]

    wire(est, n_up=1, n_down=1, phase_logpsi=phase_logpsi)

    data = _PositionData(positions=jnp.array([[2.0], [0.0]]))
    key = jax.random.key(0)
    est.init(data, key)
    stats, _ = est.evaluate_single_walker(
        params={}, data=data, prev_walker_stats={}, state=None, rngs=key
    )

    expected = 1.0 - np.exp(-4.0)
    np.testing.assert_allclose(float(stats["spin_squared"]), expected, atol=1e-6)


def test_single_electron_mode_cycles_minority_electrons_and_averages_to_full():
    """Single-electron mode cycles minority electron index and averages to full."""
    n_up, n_down = 2, 2

    def phase_logpsi(params, data, *, precision: str | None = None):
        del params, precision
        e = data["electrons"]
        return jnp.array(1.0), 0.7 * e[0, 0] + 0.3 * e[1, 0] - 0.5 * e[2, 0] - 0.2 * e[
            3, 0
        ]

    full_est = SpinSquared(mode="full")
    single_est = SpinSquared(mode="single_electron", vmap_chunk_size=2)
    wire(full_est, n_up=n_up, n_down=n_down, phase_logpsi=phase_logpsi)
    wire(single_est, n_up=n_up, n_down=n_down, phase_logpsi=phase_logpsi)

    electrons = jnp.arange(3 * 4 * 1, dtype=jnp.float32).reshape(3, 4, 1) * 0.1
    batched_data = BatchedData(_ElectronData(electrons=electrons), ["electrons"])
    key = jax.random.key(0)

    full_state = full_est.init(batched_data.unbatched_example(), key)
    assert full_state is None
    single_state = single_est.init(batched_data.unbatched_example(), key)
    assert single_state is not None
    assert single_state.shape == ()

    full_walkers, _ = full_est.evaluate_batch_walkers(
        {}, batched_data, {}, full_state, key
    )
    single_walkers_0, single_state = single_est.evaluate_batch_walkers(
        {}, batched_data, {}, single_state, key
    )
    assert single_state is not None
    assert single_state.shape == ()
    assert int(single_state) == 1
    single_walkers_1, single_state = single_est.evaluate_batch_walkers(
        {}, batched_data, {}, single_state, key
    )
    assert single_state is not None
    assert int(single_state) == 2

    # Average of single_electron over all n_min=2 steps equals full mode for each walker
    np.testing.assert_allclose(
        0.5 * (single_walkers_0["spin_squared"] + single_walkers_1["spin_squared"]),
        full_walkers["spin_squared"],
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "n_up,n_down", [(1, 1), (2, 1)], ids=["singlet_sz0", "doublet_sz_half"]
)
@pytest.mark.parametrize("mode", ["full", "single_electron"])
@pytest.mark.parametrize("complex_wf", [False, True], ids=["real_wf", "complex_wf"])
def test_spin_penalty_gradient_matches_exact_derivative(
    monkeypatch, n_up, n_down, mode, complex_wf
):
    """Spin penalty loss and gradient match w * <P_L>^2 and its exact derivative."""
    _mock_all_gather_identity(monkeypatch)
    weight = 2.5
    n_elec = n_up + n_down

    # 4 walkers of up to 3 electrons in 1D
    walkers = jnp.array(
        [
            [[0.5], [-0.2], [0.1]],
            [[1.1], [0.3], [-0.6]],
            [[-0.4], [0.8], [0.4]],
            [[0.2], [0.9], [-0.3]],
        ]
    )[:, :n_elec]
    batched_data = BatchedData(_ElectronData(electrons=walkers), ["electrons"])

    def log_psi(params, data, *, precision: str | None = None):
        del precision
        e = data["electrons"][:, 0]
        w = params["w"]
        real_part = w[0] * e[0] + w[1] * e[1] + w[2] * (e[0] - e[1]) ** 2
        if not complex_wf:
            return real_part
        imag_part = (w[0] - w[1]) * e[0] * e[1] + w[2] * (e[0] + e[1])
        return real_part + 1j * imag_part

    def phase_logpsi(params, data, *, precision: str | None = None):
        val = log_psi(params, data, precision=precision)
        if not complex_wf:
            return jnp.array(1.0), val
        return jnp.exp(1j * val.imag), val.real

    spin_est = SpinSquared(mode=mode, loss_weight=weight)
    wire(
        spin_est,
        n_up=n_up,
        n_down=n_down,
        phase_logpsi=phase_logpsi,
    )

    pipeline = EstimatorPipeline(
        {
            "spin": spin_est,
            "grads": LossGrad(clip_method="none", f_log_psi=log_psi),
        }
    )
    params = {"w": jnp.array([0.4, -0.3, 0.2])}
    state = pipeline.init(batched_data, jax.random.key(0))
    step_stats, state = pipeline.evaluate(
        params, batched_data, state, jax.random.key(1)
    )
    final_stats = pipeline.finalize_stats(
        jax.tree.map(itemgetter(None), step_stats), state
    )
    grads = final_stats["grads"]

    # Exact importance-sampled objective over fixed walkers with reference params_0.
    params_0 = jax.lax.stop_gradient(params)
    log_psi_0 = jnp.real(
        jax.vmap(log_psi, in_axes=(None, 0))(params_0, batched_data.data)
    )

    def exact_objective(p):
        lp = jnp.real(jax.vmap(log_psi, in_axes=(None, 0))(p, batched_data.data))
        weights = jnp.exp(2.0 * (lp - log_psi_0))
        weights = weights / jnp.sum(weights)
        p_locals = jnp.real(
            jax.vmap(lambda d: spin_est._compute_p_local(p, d, jnp.array(0)))(
                batched_data.data
            )
        )
        mean_p = jnp.sum(weights * p_locals)
        return weight * mean_p**2

    expected_loss = exact_objective(params)
    expected_grad = jax.grad(exact_objective)(params)

    np.testing.assert_allclose(
        final_stats["loss:spin_squared"], expected_loss, atol=1e-5
    )
    np.testing.assert_allclose(
        grads["w"],
        expected_grad["w"],
        rtol=1e-5,
        atol=1e-5,
    )
