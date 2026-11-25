# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hall workflow wiring and end-to-end runs."""

import jax
import numpy as np
import pytest
from pydantic import ValidationError

from jaqmc.app.hall import HallEvalWorkflow, HallTrainWorkflow
from jaqmc.estimator.angular_momentum import (
    SphericalAngularMomentum,
    SphericalAngularMomentumProjection,
)
from jaqmc.estimator.loss_grad import LossGrad
from jaqmc.estimator.total_energy import TotalEnergy


def _train_estimators(config):
    workflow = HallTrainWorkflow.from_config(config)
    workflow.build_runtime()
    return workflow.train_stage.estimators.estimators


def _eval_estimators(config):
    workflow = HallEvalWorkflow.from_config(config)
    workflow.build_runtime()
    return workflow.evaluation_stage.estimators.estimators


def test_angular_momentum_enabled_by_default():
    train_am = _train_estimators({})["angular_momentum"]
    eval_am = _eval_estimators({})["angular_momentum"]
    assert isinstance(train_am, SphericalAngularMomentum)
    assert train_am.loss_weight is not None
    assert isinstance(eval_am, SphericalAngularMomentum)
    assert eval_am.loss_weight is None


def test_angular_momentum_can_be_disabled_without_penalty():
    config = {"estimators": {"angular_momentum": {"enabled": False}}}
    assert "angular_momentum" not in _train_estimators(config)


def test_eval_angular_momentum_without_energy():
    estimators = _eval_estimators({"estimators": {"energy": {"enabled": False}}})
    assert "kinetic" not in estimators
    assert isinstance(estimators["angular_momentum"], SphericalAngularMomentum)


def test_penalty_configures_angular_momentum_estimator():
    estimators = _train_estimators(
        {
            "estimators": {
                "angular_momentum": {
                    "loss_weight": {
                        "lz": -1.5,
                        "lz_square": 1.0,
                        "l_square": 0.5,
                    },
                }
            },
        }
    )
    am = estimators["angular_momentum"]
    assert isinstance(am, SphericalAngularMomentum)
    assert am.loss_weight is not None
    np.testing.assert_allclose(am.loss_weight.lz, -1.5)
    np.testing.assert_allclose(am.loss_weight.lz_square, 1.0)
    np.testing.assert_allclose(am.loss_weight.l_square, 0.5)


def test_default_energy_loss_grads():
    train_est = _train_estimators({})
    total = train_est["total"]
    assert isinstance(total, TotalEnergy)
    np.testing.assert_allclose(total.loss_weight, 1.0)
    grads = train_est["loss_grad"]
    assert isinstance(grads, LossGrad)
    assert grads.clip_method == "mad"
    np.testing.assert_allclose(grads.clip_scale, 100.0)
    am = train_est["angular_momentum"]
    assert isinstance(am, SphericalAngularMomentum)
    assert am.loss_weight is not None
    np.testing.assert_allclose(am.loss_weight.lz, 0.0)
    np.testing.assert_allclose(am.loss_weight.lz_square, 0.0)
    np.testing.assert_allclose(am.loss_weight.l_square, 0.0)

    eval_est = _eval_estimators({})
    assert isinstance(eval_est["total"], TotalEnergy)
    assert eval_est["total"].loss_weight is None


def test_train_grads_override():
    grads = _train_estimators(
        {
            "train": {
                "grads": {
                    "clip_method": "none",
                    "clip_scale": 7.0,
                }
            }
        }
    )["loss_grad"]
    assert isinstance(grads, LossGrad)
    assert grads.clip_method == "none"
    np.testing.assert_allclose(grads.clip_scale, 7.0)


def test_angular_projection_is_opt_in():
    assert "angular_projection" not in _eval_estimators({})
    config = {"estimators": {"angular_projection": {"enabled": True}}}
    assert isinstance(
        _eval_estimators(config)["angular_projection"],
        SphericalAngularMomentumProjection,
    )


def test_rejects_inconsistent_estimator_config():
    config = {
        "estimators": {
            "energy": {"enabled": False},
            "angular_projection": {"enabled": True},
        }
    }
    with pytest.raises(
        ValidationError, match=r"projection requires the total energy estimator"
    ):
        HallEvalWorkflow.from_config(config)

    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        HallEvalWorkflow.from_config(
            {},
            ["estimators.angular_momentum.loss_weight.lz=1.0"],
        )

    # Disabling energy loss weight is valid when an angular-momentum penalty is active.
    _train_estimators(
        {
            "estimators": {
                "energy": {"total": {"loss_weight": 0.0}},
                "angular_momentum": {"loss_weight": {"l_square": 1.0}},
            }
        }
    )


def _run_config(tmp_path, *, wf_module: str, save_name: str):
    return {
        "workflow": {
            "save_path": str(tmp_path / save_name),
            "batch_size": 16,
            "seed": 0,
        },
        "system": {"flux": 6, "nspins": (3, 0)},
        "wf": {"module": wf_module},
    }


def _evaluation_run_config():
    return {
        "iterations": 50,
        "burn_in": 0,
        # JAX 0.5.x and 0.6.x cannot validate this Hessian path inside shard_map.
        "check_vma": jax.__version_info__[:2] not in {(0, 5), (0, 6)},
    }


@pytest.mark.parametrize("wf_module", ["laughlin", "free"])
def test_train_rejects_parameter_free_wavefunction(tmp_path, wf_module):
    config = _run_config(tmp_path, wf_module=wf_module, save_name="train")
    config["train"] = {"run": {"iterations": 1}}

    with pytest.raises(ValueError, match="no trainable parameters"):
        HallTrainWorkflow.from_config(config)()

    assert not list((tmp_path / "train").glob("train_ckpt_*.npz"))


def test_laughlin_eval_keeps_local_energy_variance_bounded(tmp_path):
    config = _run_config(tmp_path, wf_module="laughlin", save_name="eval")
    config["run"] = _evaluation_run_config()
    HallEvalWorkflow.from_config(config)()

    digest = np.load(tmp_path / "eval" / "evaluation_digest.npz")
    assert digest["energy:potential"] > 0
    assert digest["total_energy_real_var"] < 0.5


def test_free_eval_is_non_interacting(tmp_path):
    config = _run_config(tmp_path, wf_module="free", save_name="eval")
    config["system"]["interaction_strength"] = 0.0
    config["run"] = _evaluation_run_config()
    HallEvalWorkflow.from_config(config)()

    digest = np.load(tmp_path / "eval" / "evaluation_digest.npz")
    np.testing.assert_allclose(digest["energy:potential"], 0.0, atol=1e-6)
    np.testing.assert_allclose(digest["energy:kinetic"], 1.5, atol=1e-2)
    np.testing.assert_allclose(digest["total_energy_real"], 1.5, atol=1e-2)
