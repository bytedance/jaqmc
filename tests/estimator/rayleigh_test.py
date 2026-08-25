# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.data import Data
from jaqmc.estimator.rayleigh import (
    CrossLocalEnergyEvaluator,
    RayleighMatrixEstimator,
)
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.wavefunction.determinant_state import SubspaceSpec


class ToyData(Data):
    electrons: jax.Array


def test_rayleigh_estimator_matches_direct_solve():
    phi = jnp.array([[2.0, 0.5], [0.25, 1.5]], dtype=jnp.complex64)
    logs = jnp.log(phi)
    local_energy = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.complex64)
    estimator = RayleighMatrixEstimator(
        matrix_dtype="complex64",
        f_component_logpsi_matrix=lambda params, data: logs,
        f_cross_local_energy=lambda params, data, rngs: local_energy,
    )

    stats, state = estimator.evaluate_single_walker(
        {}, ToyData(electrons=jnp.zeros((2, 1, 1))), {}, None, jax.random.key(0)
    )
    expected = jnp.linalg.solve(phi, phi * local_energy)

    assert state is None
    np.testing.assert_allclose(stats["local_rayleigh"], expected, rtol=1e-6)
    np.testing.assert_allclose(
        stats["subspace_energy"], jnp.trace(expected).real, rtol=1e-6
    )
    np.testing.assert_allclose(
        stats["subspace_local_energy"], jnp.trace(expected), rtol=1e-6
    )
    # CUDA complex64 small solves have backend-dependent residuals around
    # 1e-4; the estimator's production default remains complex128.
    np.testing.assert_allclose(stats["rayleigh_solve_residual"], 0, atol=5e-4)


def test_m1_rayleigh_reduces_to_native_local_energy():
    estimator = RayleighMatrixEstimator(
        matrix_dtype="complex64",
        f_component_logpsi_matrix=lambda params, data: jnp.array([[3.0]]),
        f_cross_local_energy=lambda params, data, rngs: jnp.array([[1.25]]),
    )

    stats, _ = estimator.evaluate_single_walker(
        {}, ToyData(electrons=jnp.zeros((1, 1, 1))), {}, None, jax.random.key(0)
    )

    np.testing.assert_allclose(stats["local_rayleigh"], [[1.25]], rtol=1e-6)
    np.testing.assert_allclose(stats["subspace_energy"], 1.25, rtol=1e-6)


def test_cross_local_energy_reuses_ordered_native_estimator_pipeline():
    def component(params, data, prev_stats, state, rngs):
        del prev_stats, rngs
        value = params["slope"] * jnp.sum(data.electrons)
        return {"energy:toy": value}, state

    evaluator = CrossLocalEnergyEvaluator(
        {"component": component, "total": TotalEnergy()},
        SubspaceSpec(2),
        pair_chunk_size=2,
    )
    data = ToyData(electrons=jnp.array([[[1.0]], [[3.0]]]))
    params = {"slope": jnp.array([2.0, 5.0])}
    evaluator.init(data, jax.random.key(0))

    actual = evaluator(params, data, jax.random.key(1))

    np.testing.assert_allclose(actual, [[2.0, 5.0], [6.0, 15.0]])


def test_cross_local_energy_reuses_state_independent_potential():
    def potential(params, data, prev_stats, state, rngs):
        del params, prev_stats, rngs
        return {"energy:potential": jnp.sum(data.electrons)}, state

    def kinetic(params, data, prev_stats, state, rngs):
        del data, prev_stats, rngs
        return {"energy:kinetic": params["slope"]}, state

    evaluator = CrossLocalEnergyEvaluator(
        {"potential": potential, "kinetic": kinetic, "total": TotalEnergy()},
        SubspaceSpec(2),
        pair_chunk_size=1,
    )
    data = ToyData(electrons=jnp.array([[[1.0]], [[3.0]]]))
    params = {"slope": jnp.array([2.0, 5.0])}
    evaluator.init(data, jax.random.key(0))

    actual = evaluator(params, data, jax.random.key(1))

    np.testing.assert_allclose(actual, [[3.0, 6.0], [5.0, 8.0]])


@pytest.mark.parametrize("n_states", [2, 4, 8, 16])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_cross_local_energy_dynamic_pair_indexing_scales(n_states, chunk_size):
    def component(params, data, prev_stats, state, rngs):
        del prev_stats, rngs
        return {
            "total_energy": params["slope"] * jnp.sum(data.electrons)
        }, state

    evaluator = CrossLocalEnergyEvaluator(
        {"total": component},
        SubspaceSpec(n_states),
        pair_chunk_size=chunk_size,
    )
    data = ToyData(
        electrons=jnp.arange(1, n_states + 1, dtype=float)[:, None, None]
    )
    params = {"slope": jnp.arange(1, n_states + 1, dtype=float)}
    evaluator.init(data, jax.random.key(0))

    actual = jax.jit(evaluator)(params, data, jax.random.key(1))
    expected = jnp.arange(1, n_states + 1, dtype=float)[:, None] * jnp.arange(
        1, n_states + 1, dtype=float
    )[None, :]

    np.testing.assert_allclose(actual, expected)


def test_numerical_failure_invalidates_the_whole_step_without_filtering():
    estimator = RayleighMatrixEstimator(matrix_dtype="complex64")
    matrices = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, jnp.nan], [30.0, 40.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=jnp.complex64,
    )
    stats = {
        "local_rayleigh": matrices,
        "subspace_local_energy": jnp.array([5.0, jnp.nan, 13.0]),
        "subspace_energy": jnp.array([5.0, jnp.nan, 13.0]),
        "rayleigh_valid": jnp.array([True, True, True]),
    }

    reduced = estimator.reduce(stats)

    assert jnp.isnan(reduced["rayleigh_mean"]).any()
    np.testing.assert_allclose(reduced["rayleigh_valid_fraction"], 2 / 3)
    np.testing.assert_allclose(reduced["rayleigh_invalid_count"], 1)
    assert not reduced["training_step_valid"]


def test_subspace_energy_variance_uses_all_walkers():
    estimator = RayleighMatrixEstimator(matrix_dtype="complex64")
    matrices = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=jnp.complex64,
    )
    stats = {
        "local_rayleigh": matrices,
        "subspace_local_energy": jnp.array([5.0, 13.0], dtype=jnp.complex64),
        "subspace_energy": jnp.array([5.0, 13.0]),
        "rayleigh_valid": jnp.array([True, True]),
    }

    reduced = estimator.reduce(stats)

    np.testing.assert_allclose(reduced["subspace_energy"], 9.0)
    np.testing.assert_allclose(reduced["subspace_energy_var"], 16.0)
    assert reduced["training_step_valid"]


def test_ill_conditioned_finite_solve_remains_a_valid_sample():
    phi = jnp.array(
        [[1.0, 1.0], [1.0, 1.0001]], dtype=jnp.complex64
    )
    local_energy = jnp.array(
        [[1.0, 2.0], [3.0, 4.0]], dtype=jnp.complex64
    )
    estimator = RayleighMatrixEstimator(
        matrix_dtype="complex64",
        condition_warning=1e3,
        f_component_logpsi_matrix=lambda params, data: jnp.log(phi),
        f_cross_local_energy=lambda params, data, rngs: local_energy,
    )

    stats, _ = estimator.evaluate_single_walker(
        {}, ToyData(electrons=jnp.zeros((2, 1, 1))), {}, None, jax.random.key(0)
    )

    assert stats["amplitude_condition_warning"]
    assert stats["rayleigh_valid"]
    assert jnp.isfinite(stats["rayleigh_solve_residual"])


def test_near_singular_amplitude_matrix_triggers_diagnostic():
    logs = jnp.log(jnp.array([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.complex64))
    estimator = RayleighMatrixEstimator(
        matrix_dtype="complex64",
        f_component_logpsi_matrix=lambda params, data: logs,
        f_cross_local_energy=lambda params, data, rngs: jnp.ones((2, 2)),
    )

    stats, _ = estimator.evaluate_single_walker(
        {}, ToyData(electrons=jnp.zeros((2, 1, 1))), {}, None, jax.random.key(0)
    )

    assert stats["amplitude_condition_warning"]
    assert not stats["rayleigh_finite"]
