# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jaqmc_contrib_lit.inversion import (
    forward_lit,
    initialize_lit_poles,
    invert_signed_lit,
    lit_block_statistics,
    lit_pole_kernel,
    oscillator_strengths,
)


def test_lit_pole_kernel_is_article_discrete_line_kernel():
    omega = np.asarray([0.7, 0.8])
    eta = 0.02
    energies = np.asarray([0.75, 0.9])

    actual = lit_pole_kernel(omega, eta, energies)

    expected = 1.0 / ((energies[np.newaxis, :] - omega[:, np.newaxis]) ** 2 + eta**2)
    np.testing.assert_allclose(actual, expected)


def test_block_statistics_returns_covariance_of_mean():
    blocks = np.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            [[3.0, 2.0, 1.0, 0.0], [4.0, 3.0, 2.0, 1.0]],
        ]
    )

    stats = lit_block_statistics(blocks)

    centered = blocks - np.mean(blocks, axis=-1, keepdims=True)
    expected = np.einsum("aib,ajb->aij", centered, centered) / (4 * 3)
    np.testing.assert_allclose(stats.mean, np.mean(blocks, axis=-1))
    np.testing.assert_allclose(stats.covariance, expected)
    np.testing.assert_allclose(
        stats.standard_error,
        np.sqrt(np.diagonal(expected, axis1=-2, axis2=-1)),
    )
    assert stats.block_count == 4


def test_article_fit_recovers_shared_line_strengths_and_linear_background():
    omega = np.linspace(0.72, 0.84, 121)
    eta = 0.003
    exact_energy = np.asarray([0.78])
    exact_strengths = np.asarray([[0.18], [0.22], [0.31]])
    exact_background = np.asarray([[0.02, 0.01], [0.03, -0.02], [0.04, 0.005]])
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=exact_energy,
        pole_strengths=exact_strengths,
        background_coefficients=exact_background,
        background_center=0.78,
        background_scale=0.06,
    )

    result = invert_signed_lit(
        omega,
        eta,
        signed_lit,
        pole_energies=(0.775,),
        pole_energy_bounds=((0.75, 0.81),),
        background_order=1,
        pole_fit_tolerance=1e-10,
        pole_fit_max_iterations=300,
    )

    np.testing.assert_allclose(result.pole_energies, exact_energy, atol=1e-9)
    np.testing.assert_allclose(result.pole_strengths, exact_strengths, atol=1e-9)
    np.testing.assert_allclose(
        result.background_coefficients,
        exact_background,
        atol=1e-8,
    )
    np.testing.assert_allclose(result.fitted_lit, signed_lit, atol=1e-8)
    assert result.diagnostics.objective < 1e-15
    assert result.diagnostics.pole_fit_success
    assert not result.diagnostics.underdetermined


def test_article_fit_rejects_multiple_eta_values():
    omega = np.asarray([0.70, 0.72, 0.74, 0.76])
    eta = np.asarray([0.003, 0.003, 0.005, 0.005])
    signed_lit = np.ones_like(omega)

    with pytest.raises(ValueError, match="exactly one fixed eta"):
        invert_signed_lit(
            omega,
            eta,
            signed_lit,
            pole_energies=(0.73,),
            background_order=0,
        )


def test_initializer_uses_ordinary_line_plus_background_objective():
    omega = np.linspace(0.72, 0.84, 81)
    signed_lit = forward_lit(
        omega,
        0.003,
        pole_energies=(0.78,),
        pole_strengths=(0.2,),
        background_coefficients=(0.03,),
    )

    initialization = initialize_lit_poles(
        omega,
        0.003,
        signed_lit,
        pole_count=1,
        background_order=0,
        candidate_grid_points=257,
    )

    assert initialization.pole_energies[0] == pytest.approx(0.78, abs=5e-4)
    assert initialization.pole_energy_bounds.shape == (1, 2)
    assert initialization.objective >= 0.0


def test_oscillator_strength_uses_all_cartesian_transition_strengths():
    energies = np.asarray([0.4, 0.6])
    strengths = np.asarray(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
    )

    actual = oscillator_strengths(energies, strengths, axis_indices=(2, 0, 1))

    expected = (2.0 / 3.0) * energies * np.sum(strengths, axis=0)
    np.testing.assert_allclose(actual, expected)


def test_oscillator_strength_rejects_incomplete_cartesian_axes():
    with pytest.raises(ValueError, match="exactly x, y, and z"):
        oscillator_strengths(
            (0.4,),
            np.asarray([[0.1], [0.2]]),
            axis_indices=(0, 2),
        )
