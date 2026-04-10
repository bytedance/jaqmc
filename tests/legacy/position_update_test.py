# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import chex
import jax
import numpy as np
import pytest
import scipy as sp
from jax import numpy as jnp

from jaqmc_legacy.dmc.position_update import (
    calc_G_log,
    calc_nucleus_overshoot_prob_helper,
    clip_velocity_helper,
    do_diffusion,
    do_drift,
    sample_gamma,
)


def test_velocity_clipping_mild_v() -> None:
    velocity = jnp.array([0.3, 0.5, 0.2])
    clipped_velocity = clip_velocity_helper(
        velocity,
        jnp.array([1.0, 0, 0]),
        jnp.array([2.0, 0, 0]),
        1.0,
        1e-2,
    )
    chex.assert_trees_all_close(clipped_velocity, velocity, rtol=1e-3)


def test_velocity_clipping_outrageous_v() -> None:
    velocity = jnp.array([10, 0, 0])
    a = 1 + 100 / 1040
    v_norm = np.linalg.norm(np.asarray(velocity))
    expected_clip_factor = (-1 + np.sqrt(1 + 2 * a * v_norm**2 * 1e-1)) / (
        a * v_norm**2 * 1e-1
    )

    clipped_velocity = clip_velocity_helper(
        velocity,
        jnp.array([1.0, 0, 0]),
        jnp.array([0.0, 0, 0]),
        10.0,
        1e-1,
    )
    chex.assert_trees_all_close(clipped_velocity, expected_clip_factor * velocity)


def test_do_drift_mild() -> None:
    position = jnp.array([1.0, 0, 0])
    clipped_velocity = jnp.array([0.1, 0.2, 0.3])
    drifted_position = do_drift(
        position,
        jnp.array([2.0, 0, 0]),
        clipped_velocity,
        1e-2,
    )
    chex.assert_trees_all_close(
        drifted_position,
        position + clipped_velocity * 1e-2,
        rtol=1e-3,
    )


def test_do_drift_overshoot() -> None:
    drifted_position = do_drift(
        jnp.array([1.9, 0, 0]),
        jnp.array([2.0, 0, 0]),
        jnp.array([2.0, 2, 3]),
        1e-1,
    )
    chex.assert_trees_all_close(drifted_position, jnp.array([2.0, 0, 0]))


def test_do_drift_close_to_overshoot() -> None:
    position = jnp.array([1, 0, 0])
    clipped_velocity = jnp.array([9, 2, 3])
    drifted_position = do_drift(
        position,
        jnp.array([2.0, 0, 0]),
        clipped_velocity,
        1e-1,
    )
    expected_drift_position = (
        position
        + jnp.concatenate(
            [clipped_velocity[:1], clipped_velocity[1:] * (2 * 0.1 / 1.1)]
        )
        * 1e-1
    )
    naive_drift_position = position + clipped_velocity * 1e-1

    chex.assert_trees_all_close(drifted_position, expected_drift_position)
    assert float(jnp.linalg.norm(naive_drift_position[1:])) != pytest.approx(
        float(jnp.linalg.norm(drifted_position[1:]))
    )


def test_overshoot_prob_small() -> None:
    overshoot_prob = calc_nucleus_overshoot_prob_helper(
        jnp.array([1, 0, 0]),
        jnp.array([2.0, 0, 0]),
        jnp.array([1.0, 2, 3]),
        1e-2,
    )
    assert float(overshoot_prob) < 0.1


def test_overshoot_prob_large() -> None:
    overshoot_prob = calc_nucleus_overshoot_prob_helper(
        jnp.array([1.9, 0, 0]),
        jnp.array([2.0, 0, 0]),
        jnp.array([100.0, 2, 3]),
        1e-2,
    )
    assert float(overshoot_prob) > 0.9


def test_overshoot_prob_exact() -> None:
    overshoot_prob = calc_nucleus_overshoot_prob_helper(
        jnp.array([0.9, 0, 0]),
        jnp.array([0.0, 0, 0]),
        jnp.array([1.0, 2, 3]),
        0.1,
    )
    expected_prob = 0.5 * jax.lax.erfc(1 / jnp.sqrt(0.2))
    chex.assert_trees_all_close(overshoot_prob, expected_prob)


def test_do_diffusion_gaussian() -> None:
    key = jax.random.PRNGKey(42)
    all_keys = jax.random.split(key, 20_000)
    time_step = 1e-2
    drifted_position = jnp.array([10.0, 0, 0])
    laplace_zeta = 1 / jnp.sqrt(time_step)

    vmapped_do_diffusion = jax.vmap(
        lambda k: do_diffusion(
            drifted_position,
            jnp.array([-10.0, 0, 0]),
            0,
            k,
            time_step,
            laplace_zeta,
        )[0]
    )
    diffused_position = vmapped_do_diffusion(all_keys)
    gaussian_vars = (
        (diffused_position - drifted_position) / jnp.sqrt(time_step)
    ).reshape((-1,))

    test_result = sp.stats.kstest(np.asarray(gaussian_vars), "norm")
    assert float(test_result.pvalue) > 0.1


def _assert_gamma_like_distribution(data: jnp.ndarray, laplace_zeta: float) -> None:
    data_norm = jnp.linalg.norm(data, axis=-1)
    x, y, z = data.T
    theta = jnp.arccos(z / data_norm)
    sin_phi_sign = y / jnp.sin(theta)

    phi_between_half_pi = jnp.arctan(y / x)
    phi = (
        phi_between_half_pi
        + (phi_between_half_pi < 0) * (sin_phi_sign > 0) * jnp.pi
        - (phi_between_half_pi > 0) * (sin_phi_sign < 0) * jnp.pi
    )

    gamma_vars = data_norm * 2 * laplace_zeta
    uniform_vars = (phi + jnp.pi) / (2 * jnp.pi)

    test_result_norm = sp.stats.kstest(np.asarray(gamma_vars), "gamma", args=(3.0,))
    test_result_phi = sp.stats.kstest(np.asarray(uniform_vars), "uniform")
    assert float(test_result_norm.pvalue) > 0.01
    assert float(test_result_phi.pvalue) > 0.01

    correct_l = 0
    total_l = 0
    for ell in np.arange(0, np.pi, 0.01):
        n = np.sum(np.asarray(theta) < ell)
        if np.isclose(n / len(theta), (1 - np.cos(ell)) / 2, rtol=0.05):
            correct_l += 1
        total_l += 1
    assert correct_l / total_l > 0.95


def test_do_diffusion_gamma() -> None:
    key = jax.random.PRNGKey(42)
    all_keys = jax.random.split(key, 200_000)
    time_step = 1e-2
    nearest_nucleus = jnp.array([-10.0, 0, 0])
    laplace_zeta = 1 / jnp.sqrt(time_step)

    vmapped_do_diffusion = jax.vmap(
        lambda k: do_diffusion(
            jnp.array([10.0, 0, 0]),
            nearest_nucleus,
            1,
            k,
            time_step,
            laplace_zeta,
        )[0]
    )
    gamma_vars = vmapped_do_diffusion(all_keys) - nearest_nucleus
    _assert_gamma_like_distribution(gamma_vars, float(laplace_zeta))


def test_sample_gamma() -> None:
    key = jax.random.PRNGKey(42)
    all_keys = jax.random.split(key, 200_000)
    vmapped_sample_func = jax.vmap(sample_gamma, in_axes=(0, None))
    data = vmapped_sample_func(all_keys, 10.0)
    _assert_gamma_like_distribution(data, 10.0)


def test_calc_g_log_returns_finite_scalar() -> None:
    time_step = 0.01
    val = calc_G_log(
        jnp.zeros((3,)),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([2.0, 0.0, 0.0]),
        0.1,
        1 / jnp.sqrt(time_step),
        time_step,
    )
    assert jnp.isfinite(val)
