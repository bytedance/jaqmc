# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined, method-assign"

from typing import NamedTuple

import jax
import numpy as np
import pytest
from jaqmc_contrib_lit.config import LITConfig
from jaqmc_contrib_lit.continuation_policy import (
    _continuation_checkpoint_digests,
    _empty_lit_stats,
)
from jaqmc_contrib_lit.state import _ContinuationRecord
from jaqmc_contrib_lit.workflow import LITWorkflow
from jax import numpy as jnp


class _BridgeStats(NamedTuple):
    loss: jax.Array
    fidelity: jax.Array
    reverse_kl: jax.Array
    invalid_sample_fraction: jax.Array
    reweight_ess_fraction: jax.Array
    signed_lit: jax.Array
    source_norm: jax.Array


def _bridge_stats(fidelity: float, *, ess: float = 1.0) -> _BridgeStats:
    return _BridgeStats(
        loss=jnp.asarray(1.0 - fidelity),
        fidelity=jnp.asarray(fidelity),
        reverse_kl=jnp.asarray(0.0),
        invalid_sample_fraction=jnp.asarray(0.0),
        reweight_ess_fraction=jnp.asarray(ess),
        signed_lit=jnp.asarray(1.0),
        source_norm=jnp.asarray(1.0),
    )


def test_adaptive_continuation_bisects_and_propagates_accepted_bridges():
    workflow = object.__new__(LITWorkflow)
    workflow.lit_config = LITConfig()
    workflow.lit_config.solver.warm_start_omega = 0.0
    workflow.lit_config.continuation.iterations = 1
    workflow.lit_config.continuation.step_fraction = 10.0
    workflow.lit_config.continuation.fidelity_retention = 0.95
    workflow.lit_config.continuation.minimum_step = 0.01
    workflow.lit_config.continuation.maximum_points = 20
    optimization_starts = []

    def evaluate(*, response_params, omega, **_kwargs):
        gap = abs(float(response_params) - float(omega))
        return _bridge_stats(float(np.exp(-0.4 * gap)))

    def optimize(
        _update_step,
        response_params,
        _train_pool,
        _eval_pool,
        rng,
        *,
        omega,
        **_kwargs,
    ):
        optimization_starts.append(float(response_params))
        return jnp.asarray(omega), _bridge_stats(1.0), 1, rng

    workflow._evaluate_lit_checkpoint = evaluate
    workflow._optimize_lit_frequency = optimize

    params, _, records, _ = workflow._continue_lit_to_spectrum(
        None,
        jnp.asarray(0.0),
        _bridge_stats(1.0),
        None,
        None,
        jax.random.PRNGKey(0),
        response_apply=None,
        ground_logpsi=None,
        ground_params=None,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-2.0,
        target_omega=1.0,
        spectrum_omega=np.asarray([1.0, 1.1]),
    )

    bridges = [record for record in records if record.optimized]
    assert bridges
    assert any(record.bisections > 0 for record in bridges)
    assert np.all(np.diff([record.omega for record in bridges]) > 0.0)
    assert optimization_starts == pytest.approx(
        [0.0, *[record.omega for record in bridges[:-1]]]
    )
    assert float(params) == pytest.approx(bridges[-1].omega)
    assert records[-1].omega == pytest.approx(1.0)
    assert not records[-1].optimized


def _checkpoint_inputs(config: LITConfig):
    response_params = {"w": jnp.asarray([1.0 + 2.0j])}
    ground_params = {"g": jnp.asarray([3.0])}
    digest_args = dict(
        response_params=response_params,
        ground_params=ground_params,
        train_pool={"electrons": jnp.asarray([[0.1, 0.2, 0.3]])},
        eval_pool={"electrons": jnp.asarray([[0.4, 0.5, 0.6]])},
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-2.0,
        ground_checkpoint_step=7,
        response_parity=-1,
        target_omega=0.4,
        spectrum_omega=np.asarray([0.4, 0.5]),
    )
    digests = _continuation_checkpoint_digests(config, **digest_args)
    return response_params, ground_params, digest_args, digests


def test_continuation_checkpoint_round_trip_revalidates_for_resume(tmp_path):
    old_run = tmp_path / "old"
    config = LITConfig()
    config.solver.warm_start_omega = 0.0
    config.continuation.ess_fraction_minimum = 0.05
    response_params, ground_params, _, digests = _checkpoint_inputs(config)
    state_fingerprint, full_digest = digests
    stats = _empty_lit_stats()._replace(
        loss=jnp.asarray(0.005),
        fidelity=jnp.asarray(0.995),
        reverse_kl=jnp.asarray(0.002),
        signed_lit=jnp.asarray(1.0),
        source_norm=jnp.asarray(1.0),
        reweight_ess_fraction=jnp.asarray(0.8),
        invalid_sample_fraction=jnp.asarray(0.0),
    )
    record = _ContinuationRecord(
        omega=0.2,
        optimized=True,
        selected_iteration=100,
        stats=stats,
        inherited_fidelity=0.99,
        step=0.2,
        bisections=0,
        probe_accepted=True,
        min_step_override=False,
    )
    rng = jax.random.PRNGKey(17)
    saver = object.__new__(LITWorkflow)
    saver.lit_config = config
    saver.save_path = old_run
    saver._save_lit_continuation_checkpoint(
        response_params,
        stats,
        rng,
        0.2,
        [record],
        axis=0,
        target_omega=0.4,
        ground_checkpoint_step=7,
        ground_energy=-2.0,
        source_center=0.0,
        source_norm=1.0,
        response_parity=-1,
        state_fingerprint=state_fingerprint,
        full_config_digest=full_digest,
        warm_start_selected_iteration=1500,
    )

    restore_config = LITConfig()
    restore_config.solver.warm_start_omega = 0.0
    restore_config.continuation.ess_fraction_minimum = 0.05
    restore_config.continuation.restore_path = str(old_run)
    _, _, _, restore_digests = _checkpoint_inputs(restore_config)
    workflow = object.__new__(LITWorkflow)
    workflow.lit_config = restore_config
    workflow.save_path = tmp_path / "new"
    revalidated = stats._replace(fidelity=jnp.asarray(0.996))
    workflow._evaluate_lit_checkpoint = lambda **_kwargs: revalidated

    restored = workflow._restore_lit_continuation_checkpoint(
        {"w": jnp.asarray([0.0 + 0.0j])},
        jax.random.PRNGKey(0),
        None,
        response_apply=None,
        ground_logpsi=None,
        ground_params=ground_params,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-2.0,
        ground_checkpoint_step=7,
        response_parity=-1,
        target_omega=0.4,
        state_fingerprint=restore_digests[0],
        full_config_digest=restore_digests[1],
    )

    assert restored is not None
    np.testing.assert_allclose(restored.response_params["w"], [1.0 + 2.0j])
    np.testing.assert_array_equal(restored.rng, rng)
    assert restored.current_omega == pytest.approx(0.2)
    assert float(restored.current_stats.fidelity) == pytest.approx(0.996)
    assert restored.warm_start_selected_iteration == 1500
    assert len(restored.records) == 1


def test_continuation_fingerprint_ignores_policy_but_binds_ansatz_and_pool():
    baseline = LITConfig()
    _, _, digest_args, baseline_digests = _checkpoint_inputs(baseline)
    policy_change = LITConfig()
    policy_change.solver.plateau_patience_iterations = 500
    ansatz_change = LITConfig()
    ansatz_change.ansatz.determinants = 32

    policy_digests = _continuation_checkpoint_digests(policy_change, **digest_args)
    ansatz_digests = _continuation_checkpoint_digests(ansatz_change, **digest_args)
    changed_pool_digests = _continuation_checkpoint_digests(
        baseline,
        **{
            **digest_args,
            "eval_pool": {"electrons": jnp.asarray([[999.0]])},
        },
    )

    assert policy_digests[0] == baseline_digests[0]
    assert policy_digests[1] != baseline_digests[1]
    assert ansatz_digests[0] != baseline_digests[0]
    assert changed_pool_digests[0] != baseline_digests[0]
