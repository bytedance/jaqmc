# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

from types import SimpleNamespace

import jax
import numpy as np
from jaqmc_contrib_lit.config import LITConfig
from jaqmc_contrib_lit.optimization import (
    _source_distillation_stats_from_log_ratios,
    _spring_direction_chunked,
)
from jaqmc_contrib_lit.state import _SpringState
from jaqmc_contrib_lit.workflow import LITWorkflow
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.data import BatchedData


def test_source_distillation_objective_is_scale_invariant_at_exact_source():
    source_weight = jnp.ones(4, dtype=jnp.float32)
    exact = _source_distillation_stats_from_log_ratios(
        jnp.full(4, 37.0 + 0.6j, dtype=jnp.complex64),
        source_weight,
        reverse_kl_weight=1.0,
    )
    shifted = _source_distillation_stats_from_log_ratios(
        jnp.full(4, -51.0 - 2.2j, dtype=jnp.complex64),
        source_weight,
        reverse_kl_weight=1.0,
    )

    np.testing.assert_allclose(float(exact.fidelity), 1.0, rtol=2e-6)
    np.testing.assert_allclose(float(exact.reverse_kl), 0.0, atol=2e-6)
    np.testing.assert_allclose(float(exact.reweight_ess_fraction), 1.0, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(exact), atol=2e-6)


def test_source_distillation_selects_improved_independent_heldout_state():
    workflow = object.__new__(LITWorkflow)
    workflow.config = SimpleNamespace(batch_size=8, seed=7)
    workflow.lit_config = LITConfig()
    workflow.lit_config.source.distillation_iterations = 20
    workflow.lit_config.parallel.train_batch_size = 8
    workflow.lit_config.parallel.eval_batch_size = 8
    workflow.lit_config.solver.selection_interval = 1
    workflow.lit_config.solver.log_interval = 0
    workflow.lit_config.solver.reverse_kl_weight = 0.1
    workflow.lit_config.solver.learning_rate = 0.05
    workflow.lit_config.solver.spring_epsilon = 0.01
    workflow.lit_config.solver.spring_decay = 0.0
    workflow.lit_config.solver.spring_damping_floor = 1e-6
    workflow.lit_config.solver.sr_max_norm = 0.1

    def pool(source_coordinates):
        electrons = jnp.stack(
            (
                -jnp.asarray(source_coordinates, dtype=jnp.float32),
                jnp.zeros(len(source_coordinates), dtype=jnp.float32),
                jnp.zeros(len(source_coordinates), dtype=jnp.float32),
            ),
            axis=1,
        )[:, None, :]
        return BatchedData(
            data=MoleculeData(
                electrons=electrons,
                atoms=jnp.zeros((1, 3), dtype=jnp.float32),
                charges=jnp.ones(1, dtype=jnp.float32),
            ),
            fields_with_batch=("electrons",),
        )

    train_pool = pool([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9])
    eval_pool = pool([-0.85, -0.65, -0.45, -0.25, 0.25, 0.45, 0.65, 0.85])

    def ground_logpsi(_params, _data):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def response_apply(params, data):
        source = -data.electrons[0, 0]
        amplitude = params["bias"] + params["slope"] * source
        return jnp.log(amplitude.astype(jnp.complex64))

    initial_params = {
        "bias": jnp.asarray(1.0, dtype=jnp.float32),
        "slope": jnp.asarray(0.2, dtype=jnp.float32),
    }
    common = dict(
        response_apply=response_apply,
        ground_logpsi=ground_logpsi,
        ground_params={},
        eval_pool=eval_pool,
        axis=0,
        source_center=0.0,
    )
    initial_stats = workflow._evaluate_source_distillation(
        response_params=initial_params,
        **common,
    )
    selected_params, _ = workflow._distill_response_from_source(
        response_apply,
        initial_params,
        ground_logpsi,
        {},
        train_pool,
        eval_pool,
        jax.random.PRNGKey(3),
        axis=0,
        source_center=0.0,
    )
    selected_stats = workflow._evaluate_source_distillation(
        response_params=selected_params,
        **common,
    )

    assert float(selected_stats.loss) < float(initial_stats.loss)
    assert float(selected_stats.fidelity) > float(initial_stats.fidelity) + 0.1


def test_spring_resets_history_when_the_metric_has_zero_mass():
    score = jnp.zeros((4, 2), dtype=jnp.float32)
    state = _SpringState(previous_direction=jnp.asarray([3.0, -2.0], dtype=jnp.float32))

    direction, next_state, damping = _spring_direction_chunked(
        (4,),
        lambda _: score,
        jnp.zeros(2, dtype=jnp.float32),
        state,
        epsilon_scale=1e-3,
        damping_floor=1e-12,
        decay=0.99,
    )

    assert np.isfinite(float(damping))
    np.testing.assert_array_equal(direction, np.zeros(2))
    np.testing.assert_array_equal(next_state.previous_direction, np.zeros(2))
