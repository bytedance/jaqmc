# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any, Literal

import jax
import numpy as np
import yaml
from jax import lax
from jax import numpy as jnp

from jaqmc.data import BatchedData, Data
from jaqmc.estimator.loss_grad import LossGrad, loss_term
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.utils.clip import clip_observable
from jaqmc.utils.config import config_to_yaml, configurable_dataclass, parse_config


class _DummyData(Data):
    g: jnp.ndarray


def _eval_loss_grad_with_mock_logpsi(
    per_walker_grads: jnp.ndarray,
    walker_stats: dict[str, object],
    *,
    clip_method: Literal["iqr", "mad", "none"] = "none",
    clip_scale: float = 5.0,
) -> dict[str, Any]:
    batched = BatchedData(_DummyData(g=per_walker_grads), ["g"])
    loss_grad = LossGrad(
        clip_method=clip_method,
        clip_scale=clip_scale,
        f_log_psi=lambda p, d, *, precision=None: p["w"] * d.g,
    )
    grad_walkers, _ = loss_grad.evaluate_batch_walkers(
        {"w": jnp.array(1.0)}, batched, walker_stats, None, jax.random.key(0)
    )
    return loss_grad.reduce(grad_walkers)


def _mock_all_gather_identity(monkeypatch):
    monkeypatch.setattr(
        lax,
        "all_gather",
        lambda x, axis_name, tiled=True: x,
    )


def test_clip_observable_iqr(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    x = jnp.array([0.0, 1.0, 2.0, 100.0])
    q1 = jnp.nanquantile(x, 0.25)
    q3 = jnp.nanquantile(x, 0.75)
    expected = jnp.clip(x, q1 - (q3 - q1), q3 + (q3 - q1))

    clipped = clip_observable(x, "iqr", scale=1.0)

    np.testing.assert_allclose(clipped, expected)


def test_clip_observable_mad_complex(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    x = jnp.array([0.0 + 100.0j, 1.0 + 1.0j, 2.0 + 2.0j, 100.0 + 0.0j])
    real = x.real
    imag = x.imag
    real_median = jnp.nanmedian(real)
    imag_median = jnp.nanmedian(imag)
    real_absdev = jnp.nanmedian(jnp.abs(real - real_median))
    imag_absdev = jnp.nanmedian(jnp.abs(imag - imag_median))
    expected = jnp.clip(
        real, real_median - real_absdev, real_median + real_absdev
    ) + 1j * jnp.clip(imag, imag_median - imag_absdev, imag_median + imag_absdev)

    clipped = clip_observable(x, "mad", scale=1.0)

    np.testing.assert_allclose(clipped, expected)


def test_clip_observable_none_is_noop(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    x = jnp.array([0.0, 1.0, 2.0, 100.0])

    clipped = clip_observable(x, "none", scale=0.01)

    np.testing.assert_allclose(clipped, x)


def test_loss_term_and_loss_grad_uses_selected_clip_method(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    losses = jnp.array([0.0, 1.0, 2.0, 100.0])
    grads_w = jnp.array([1.0, 2.0, 3.0, 4.0])
    expected_clipped = clip_observable(losses, "mad", scale=1.0)
    expected_grad = 2.0 * (
        jnp.mean(grads_w * expected_clipped)
        - jnp.mean(grads_w) * jnp.mean(expected_clipped)
    )

    step_stats = _eval_loss_grad_with_mock_logpsi(
        grads_w,
        loss_term("total_energy", losses, weight=1.0),
        clip_method="mad",
        clip_scale=1.0,
    )

    np.testing.assert_allclose(step_stats["loss:total_energy"], jnp.mean(losses))
    np.testing.assert_allclose(step_stats["loss"], jnp.mean(losses))
    np.testing.assert_allclose(step_stats["grads"]["w"], expected_grad)


def test_loss_grad_nan_validity_masking(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    # Walker 0 has NaN energy; walkers 1 and 2 are valid across all terms.
    energy = jnp.array([jnp.nan, 1.0, 3.0])
    penalty = jnp.array([2.0, 4.0, 6.0])
    grads_w = jnp.array([10.0, 2.0, 4.0])

    step_stats = _eval_loss_grad_with_mock_logpsi(
        grads_w,
        {
            **loss_term("total_energy", energy, weight=1.0),
            **loss_term("penalty", penalty, weight=1.0),
            **loss_term("disabled", energy, weight=0.0),
        },
        clip_method="none",
    )

    valid_total = jnp.array([1.0 + 4.0, 3.0 + 6.0])
    valid_grads = jnp.array([2.0, 4.0])
    expected_grad = 2.0 * jnp.mean(valid_grads * (valid_total - jnp.mean(valid_total)))

    np.testing.assert_allclose(step_stats["loss:total_energy"], 2.0)
    np.testing.assert_allclose(step_stats["loss:penalty"], 4.0)
    np.testing.assert_allclose(step_stats["grads"]["w"], expected_grad)
    assert "loss:disabled" not in step_stats


def test_total_energy_complex_gradient(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    losses = jnp.array([0j, 2.0j])
    estimator = TotalEnergy(loss_weight=1.0)
    batched = BatchedData(_DummyData(g=jnp.array([0j, 2.0j])), ["g"])

    walker_stats, _ = estimator.evaluate_batch_walkers(
        {}, batched, {"energy:kinetic": losses}, None, jax.random.key(0)
    )
    step_stats = estimator.reduce(walker_stats)
    loss_stats = _eval_loss_grad_with_mock_logpsi(
        jnp.array([0j, 2.0j]), walker_stats, clip_method="none"
    )

    assert set(step_stats) == {
        "total_energy",
        "total_energy_real",
        "total_energy_real_var",
    }
    # 2 * (mean(Re[conj(O) E]) - Re[conj(mean O) mean E]) = 2 * (2 - 1).
    np.testing.assert_allclose(loss_stats["grads"]["w"], 2.0)


def test_loss_term_combines_local_extra_and_loss_fn(monkeypatch):
    """loss_term and LossGrad combine covariance, direct_grad, and loss_fn."""
    _mock_all_gather_identity(monkeypatch)
    spin_local = jnp.array([0.0, 1.0, 2.0, 100.0])
    grads_w = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Walkers 0 and 3 are clipped outliers; their direct_grad should be zeroed.
    extra = {"w": jnp.array([0.5, 0.5, 0.5, 1e6])}
    expected_clipped = clip_observable(spin_local, "mad", scale=1.0)

    step_stats = _eval_loss_grad_with_mock_logpsi(
        grads_w,
        loss_term(
            "spin_squared",
            spin_local,
            weight=1.0,
            direct_grad=extra,
            loss_fn=jnp.square,
        ),
        clip_method="mad",
        clip_scale=1.0,
    )

    mean_local = jnp.mean(spin_local)
    mean_clipped = jnp.mean(expected_clipped)
    np.testing.assert_allclose(step_stats["loss:spin_squared"], mean_local**2)
    np.testing.assert_allclose(step_stats["loss"], mean_local**2)

    expected_cov = 2.0 * jnp.mean(grads_w * (expected_clipped - mean_clipped))
    expected_extra = jnp.mean(jnp.array([0.0, 0.5, 0.5, 0.0]))
    np.testing.assert_allclose(
        step_stats["grads"]["w"],
        2.0 * mean_local * (expected_cov + expected_extra),
    )


def test_loss_grad_config_roundtrip():
    @configurable_dataclass
    class GradsConfig:
        grads: LossGrad = field(default_factory=LossGrad)

    grads = parse_config(
        GradsConfig,
        {
            "grads": {
                "clip_method": "iqr",
                "clip_scale": 7.0,
            }
        },
    ).grads

    assert grads.clip_method == "iqr"
    np.testing.assert_allclose(grads.clip_scale, 7.0)

    grads2 = parse_config(
        GradsConfig, yaml.safe_load(config_to_yaml(GradsConfig(grads=grads)))
    ).grads

    assert grads2.clip_method == "iqr"
    np.testing.assert_allclose(grads2.clip_scale, 7.0)


def test_loss_grad_computes_empty_without_evaluating_f_log_psi():
    class _XData(Data):
        x: jnp.ndarray

    def f_log_psi(params, data, *, precision: str | None = None):
        raise AssertionError(
            "f_log_psi should not be called when active loss terms are empty"
        )

    loss_grad = LossGrad(f_log_psi=f_log_psi)
    grad_walkers, _ = loss_grad.evaluate_batch_walkers(
        {"w": jnp.array(2.0)},
        BatchedData(_XData(x=jnp.array([3.0, 5.0])), ["x"]),
        loss_term("disabled", jnp.array([1.0, 2.0]), weight=0.0),
        None,
        jax.random.key(0),
    )
    assert grad_walkers == {}
    assert loss_grad.reduce(grad_walkers) == {}


def test_vmc_stage_wires_loss_grad_and_skips_unused_logpsi_vjp():
    from jaqmc.app.hydrogen_atom import HydrogenAtom, PotentialEnergy, data_init
    from jaqmc.estimator.base import PerWalkerEstimator
    from jaqmc.optimizer.optax import adam
    from jaqmc.sampler.mcmc import MCMCSampler
    from jaqmc.utils import parallel_jax
    from jaqmc.workflow.base import init_batched_data
    from jaqmc.workflow.stage.vmc import VMCWorkStage, VMCWorkStageConfig

    @configurable_dataclass
    class _DirectGradEstimator(PerWalkerEstimator):
        def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
            del data, prev_walker_stats, rngs
            return {
                "aux:loss:pretrain": jnp.array(1.5),
                "aux:direct_grad:pretrain": jax.tree.map(jnp.ones_like, params),
            }, state

    wf = HydrogenAtom()
    rng = jax.random.key(0)
    batched_data = init_batched_data(data_init, 4, rng)
    sub_rngs = jax.device_put(
        jax.random.split(rng, jax.device_count()),
        parallel_jax.make_sharding(parallel_jax.DATA_PARTITION),
    )

    def build_and_jit(estimators, f_log_psi):
        builder = VMCWorkStage.builder(VMCWorkStageConfig(), wf, name="train")
        builder.configure_sample_plan(wf.evaluate, {"electrons": MCMCSampler()})
        builder.configure_optimizer(adam(), f_log_psi=f_log_psi)
        builder.configure_estimators(estimators)
        builder.configure_writers({})
        stage = builder.build()
        state = stage.create_state(rng, batched_data=batched_data)
        compute = parallel_jax.jit_sharded(
            stage.compute_step,
            in_specs=(state.partition(), parallel_jax.DATA_PARTITION),
            out_specs=(state.partition(), parallel_jax.SHARE_PARTITION),
            check_vma=True,
        )
        return stage, state, compute

    # 1. Direct-gradient estimator alone: LossGrad skips the logpsi VJP so
    #    f_log_psi is not traced.
    pretrain_stage, pretrain_state, pretrain_compute = build_and_jit(
        {"grads": _DirectGradEstimator()},
        f_log_psi=lambda p, d, *, precision=None: jnp.sum(
            jnp.arctan(d.electrons * p["params"]["alpha"])
        ),
    )
    assert isinstance(pretrain_stage.estimators.estimators["loss_grad"], LossGrad)
    assert "atan" not in pretrain_compute.lower(pretrain_state, sub_rngs).as_text()
    _, pretrain_stats = pretrain_compute(pretrain_state, sub_rngs)
    np.testing.assert_allclose(pretrain_stats["loss"], 1.5)
    np.testing.assert_allclose(pretrain_stats["loss:pretrain"], 1.5)

    # 2. Loss-weighted estimators: LossGrad is wired automatically and
    #    computes loss/grads.
    train_stage, train_state, train_compute = build_and_jit(
        {
            "potential": PotentialEnergy(),
            "total": TotalEnergy(),
        },
        f_log_psi=wf.evaluate,
    )
    assert isinstance(train_stage.estimators.estimators["loss_grad"], LossGrad)
    _, stats = train_compute(train_state, sub_rngs)
    np.testing.assert_allclose(stats["loss"], stats["loss:total_energy"])
