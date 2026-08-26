# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
import yaml
from jax import lax
from jax import numpy as jnp

from jaqmc.app.hall import HallTrainWorkflow
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.loss_grad import LossAndGrad, StreamingLossAndGrad
from jaqmc.utils.clip import clip_observable
from jaqmc.utils.config import ConfigManager


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


def test_loss_and_grad_reduce_uses_selected_clip_method(monkeypatch):
    _mock_all_gather_identity(monkeypatch)
    losses = jnp.array([0.0, 1.0, 2.0, 100.0])
    grads = {"w": jnp.array([1.0, 2.0, 3.0, 4.0])}
    estimator = LossAndGrad(clip_method="mad", clip_scale=1.0)
    expected_clipped = clip_observable(losses, "mad", scale=1.0)

    reduced = estimator.reduce({"loss": losses, "grad_logpsi": grads})

    np.testing.assert_allclose(reduced["loss"], jnp.mean(losses))
    np.testing.assert_allclose(reduced["clipped_loss"], jnp.mean(expected_clipped))
    np.testing.assert_allclose(
        reduced["grad_logpsi_and_loss"]["w"],
        jnp.mean(grads["w"] * expected_clipped),
    )
    assert not any(key.endswith("_var") for key in reduced)


class GradientData(Data):
    x: jax.Array


@pytest.mark.parametrize("clip_method", ["none", "mad", "iqr"])
@pytest.mark.parametrize("complex_output", [False, True])
def test_streaming_loss_grad_matches_reference(
    monkeypatch, clip_method, complex_output
):
    _mock_all_gather_identity(monkeypatch)
    data = BatchedData(GradientData(x=jnp.array([0.2, -0.4, 0.7, 1.1, -0.8])), ["x"])
    params = {"a": jnp.array(0.6), "b": jnp.array(-0.3)}
    local_energy = jnp.array(
        [1.0 + 0.2j, -0.5 + 0.7j, 2.0 - 0.3j, 0.1 + 0.9j, 1.3 - 0.4j]
    )

    def logpsi(p, one_data):
        value = p["a"] * one_data.x + p["b"] * one_data.x**2
        return value + 1j * p["b"] * one_data.x if complex_output else value

    reference = LossAndGrad(
        loss_key="energy", clip_method=clip_method, f_log_psi=logpsi
    )
    streaming = StreamingLossAndGrad(
        loss_key="energy",
        vmap_chunk_size=2,
        clip_method=clip_method,
        f_log_psi=logpsi,
    )
    prev_stats = {"energy": local_energy}

    ref_walkers, _ = reference.evaluate_batch_walkers(
        params, data, prev_stats, None, jax.random.key(0)
    )
    ref_reduced = reference.reduce(ref_walkers)
    ref_final = reference.finalize_stats(
        jax.tree.map(lambda x: x[None], ref_reduced), None
    )
    stream_sums, _ = streaming.evaluate_batch_walkers(
        params, data, prev_stats, None, jax.random.key(0)
    )
    stream_reduced = streaming.reduce(stream_sums)
    stream_final = streaming.finalize_stats(
        jax.tree.map(lambda x: x[None], stream_reduced), None
    )

    np.testing.assert_allclose(stream_final["loss"], ref_final["loss"], rtol=1e-6)
    for actual, expected in zip(
        jax.tree.leaves(stream_final["grads"]),
        jax.tree.leaves(ref_final["grads"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_loss_and_grad_keeps_imaginary_local_energy_contribution():
    estimator = LossAndGrad(clip_method="none")
    local_energy = jnp.array([1.0 + 2.0j, 3.0 - 1.0j])
    grad_logpsi = {"w": jnp.array([1.0 + 1.0j, 2.0 - 3.0j])}

    reduced = estimator.reduce(
        {"loss": local_energy, "grad_logpsi": grad_logpsi}
    )
    final = estimator.finalize_stats(
        jax.tree.map(lambda x: x[None], reduced), None
    )
    centered = local_energy - jnp.mean(local_energy)
    expected = 2 * jnp.mean(jnp.real(jnp.conj(grad_logpsi["w"]) * centered))
    wrong_real_only = 2 * jnp.mean(
        jnp.real(jnp.conj(grad_logpsi["w"]) * jnp.real(centered))
    )

    np.testing.assert_allclose(final["grads"]["w"], expected)
    assert not np.isclose(float(expected), float(wrong_real_only))


def test_loss_and_grad_validity_mask_uses_same_walker_subset():
    estimator = LossAndGrad(clip_method="none", validity_key="valid")
    reduced = estimator.reduce(
        {
            "loss": jnp.array([1.0, 1000.0, 5.0]),
            "grad_logpsi": {"w": jnp.array([1.0, 100.0, 3.0])},
            "loss_valid": jnp.array([True, False, True]),
        }
    )
    final = estimator.finalize_stats(
        jax.tree.map(lambda x: x[None], reduced), None
    )
    expected = 2 * jnp.mean(jnp.array([1.0, 3.0]) * (jnp.array([1.0, 5.0]) - 3.0))

    np.testing.assert_allclose(final["grads"]["w"], expected)


def test_loss_and_grad_config_roundtrip_preserves_clip_method():
    cfg = ConfigManager({"grads": {"clip_method": "mad", "clip_scale": 7.0}})
    grads = cfg.get("grads", LossAndGrad())

    assert grads.clip_method == "mad"
    np.testing.assert_allclose(grads.clip_scale, 7.0)

    cfg2 = ConfigManager(yaml.safe_load(cfg.to_yaml()))
    grads2 = cfg2.get("grads", LossAndGrad())

    assert grads2.clip_method == "mad"
    np.testing.assert_allclose(grads2.clip_scale, 7.0)
    cfg2.finalize()


def test_hall_workflow_default_loss_grads():
    workflow = HallTrainWorkflow(ConfigManager({}))
    grads = workflow.train_stage.estimators.estimators["grads"]

    assert isinstance(grads, LossAndGrad)
    assert grads.loss_key == "total_energy"
    assert grads.clip_method == "iqr"
    assert grads.clip_scale == 100


def test_hall_workflow_penalty_default_loss_key():
    workflow = HallTrainWorkflow(ConfigManager({"system": {"lz_penalty": 1.0}}))
    grads = workflow.train_stage.estimators.estimators["grads"]

    assert isinstance(grads, LossAndGrad)
    assert grads.loss_key == "penalized_loss"


def test_hall_workflow_train_grads_override():
    workflow = HallTrainWorkflow(
        ConfigManager(
            {
                "train": {
                    "grads": {
                        "clip_method": "none",
                        "clip_scale": 7.0,
                    }
                }
            }
        )
    )
    grads = workflow.train_stage.estimators.estimators["grads"]

    assert isinstance(grads, LossAndGrad)
    assert grads.clip_method == "none"
    np.testing.assert_allclose(grads.clip_scale, 7.0)
