# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.utils.func_transform import grad_maybe_complex
from jaqmc.utils.subspace_linalg import stable_complex_logdet


def _synthetic_logs(theta):
    n_states = theta.shape[0]
    rows = jnp.arange(1, n_states + 1, dtype=theta.dtype)[:, None]
    cols = jnp.arange(1, n_states + 1, dtype=theta.dtype)[None, :]
    base = 0.07 * rows * cols + 0.03j * (rows - cols)
    return base + theta[None, :] * (0.2 * rows + 0.05j * rows**2)


@pytest.mark.parametrize("n_states", [2, 3])
def test_logdet_score_matches_gvmc_vjp_identity(n_states):
    theta = jnp.linspace(-0.3, 0.4, n_states, dtype=jnp.float64)
    score = grad_maybe_complex(
        lambda value: stable_complex_logdet(_synthetic_logs(value))
    )(theta)

    logs = _synthetic_logs(theta)
    phi = jnp.exp(logs)
    cotangent = phi * jnp.linalg.inv(phi).T
    _, vjp = jax.vjp(_synthetic_logs, theta)
    (score_real,) = vjp(cotangent)
    (score_imag,) = vjp(-1j * cotangent)
    reference = jax.lax.complex(score_real, score_imag)

    np.testing.assert_allclose(score, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n_states", [2, 3])
def test_stable_row_scaling_preserves_logdet_score(n_states):
    theta = jnp.linspace(-0.2, 0.5, n_states, dtype=jnp.float64)

    def direct(value):
        sign, logabs = jnp.linalg.slogdet(jnp.exp(_synthetic_logs(value)))
        return logabs + jnp.log(sign)

    stable_score = grad_maybe_complex(
        lambda value: stable_complex_logdet(_synthetic_logs(value))
    )(theta)
    direct_score = grad_maybe_complex(direct)(theta)

    np.testing.assert_allclose(stable_score, direct_score, rtol=1e-10, atol=1e-12)


def test_m1_grassmann_qgt_reduces_to_ordinary_score_covariance():
    theta = jnp.array([0.35], dtype=jnp.float64)
    samples = jnp.array([-1.0, -0.2, 0.7, 1.5], dtype=jnp.float64)

    def component(value, sample):
        return value[0] * sample + 0.1j * value[0] * sample**2

    def determinant(value, sample):
        return stable_complex_logdet(component(value, sample)[None, None])

    component_scores = jax.vmap(
        lambda sample: grad_maybe_complex(component)(theta, sample)
    )(samples)
    determinant_scores = jax.vmap(
        lambda sample: grad_maybe_complex(determinant)(theta, sample)
    )(samples)
    centered = component_scores - jnp.mean(component_scores, axis=0)
    ordinary_qgt = jnp.real(jnp.conj(centered).T @ centered) / len(samples)
    det_centered = determinant_scores - jnp.mean(determinant_scores, axis=0)
    grassmann_qgt = jnp.real(jnp.conj(det_centered).T @ det_centered) / len(samples)

    np.testing.assert_allclose(
        determinant_scores, component_scores, rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(grassmann_qgt, ordinary_qgt, rtol=1e-11)
