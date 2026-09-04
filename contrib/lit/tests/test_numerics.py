# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jaqmc_contrib_lit.optimization import (
    _lit_error_monitor,
    _solve_sr_direction_chunked,
)
from jaqmc_contrib_lit.transform import broadened_from_lit, lit_from_poles
from jax import numpy as jnp


def test_discrete_line_lit_matches_literal_lorentzian_sum():
    omega = np.asarray([0.35, 0.375, 0.40])
    eta = 0.02
    energies = np.asarray([0.375, 4.0 / 9.0, 15.0 / 32.0])
    strengths = np.asarray([0.55, 0.12, 0.04])

    actual = broadened_from_lit(
        lit_from_poles(omega, energies, strengths, eta),
        eta,
    )
    expected = (
        eta
        / np.pi
        * np.sum(
            strengths[np.newaxis, :]
            / ((energies[np.newaxis, :] - omega[:, np.newaxis]) ** 2 + eta**2),
            axis=1,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_lit_error_monitor_is_invariant_to_response_rescaling():
    expected = (
        0.125 * np.sqrt(2.25) / (0.03 * abs(0.3 - 0.4j)) * np.sqrt((1.0 - 0.91) / 0.91)
    )
    for magnitude in (1.0, 1e-12, 1e12):
        actual = _lit_error_monitor(
            fidelity=0.91,
            source_norm=2.25,
            normalization=magnitude * (0.3 - 0.4j),
            eta=0.03,
            error_d=magnitude * 0.125,
            error_d_valid=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14)


@pytest.mark.parametrize(
    ("normalization", "error_d", "valid"),
    [
        (0.0 + 0.0j, 0.2, True),
        (np.nan + 0.0j, 0.2, True),
        (0.5 + 0.0j, np.nan, True),
        (0.5 + 0.0j, 0.2, False),
    ],
)
def test_lit_error_monitor_marks_invalid_inputs_as_nan(
    normalization,
    error_d,
    valid,
):
    assert np.isnan(
        _lit_error_monitor(
            fidelity=0.9,
            source_norm=1.0,
            normalization=normalization,
            eta=0.02,
            error_d=error_d,
            error_d_valid=valid,
        )
    )


@pytest.mark.parametrize(("samples", "parameters"), [(10, 3), (4, 6)])
def test_chunked_sr_matches_dense_linear_system(samples, parameters):
    score = jnp.asarray(
        np.arange(samples * parameters, dtype=np.float32).reshape(samples, parameters)
        / 17.0
    )
    grad = jnp.linspace(-0.4, 0.7, parameters)
    damping = jnp.asarray(0.05, dtype=jnp.float32)

    dense = _solve_sr_direction_chunked(
        (samples,),
        lambda _: score,
        grad,
        damping,
    )
    split = samples // 2
    chunks = (score[:split], score[split:])
    chunked = _solve_sr_direction_chunked(
        tuple(chunk.shape[0] for chunk in chunks),
        lambda index: chunks[index],
        grad,
        damping,
    )
    np.testing.assert_allclose(chunked, dense, rtol=5e-5, atol=5e-6)
