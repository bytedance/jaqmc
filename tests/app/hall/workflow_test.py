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
from jaqmc.estimator.loss_grad import LossAndGrad


def _train_estimators(config):
    workflow = HallTrainWorkflow.from_config(config)
    workflow.build_runtime()
    return workflow.train_stage.estimators.estimators


def _eval_estimators(config):
    workflow = HallEvalWorkflow.from_config(config)
    workflow.build_runtime()
    return workflow.evaluation_stage.estimators.estimators


def test_angular_momentum_enabled_by_default():
    assert isinstance(
        _train_estimators({})["angular_momentum"], SphericalAngularMomentum
    )


def test_angular_momentum_can_be_disabled_without_penalty():
    config = {"estimators": {"angular_momentum": {"enabled": False}}}
    assert "angular_momentum" not in _train_estimators(config)


def test_eval_angular_momentum_without_energy():
    estimators = _eval_estimators({"estimators": {"energy": {"enabled": False}}})
    assert "kinetic" not in estimators
    assert isinstance(estimators["angular_momentum"], SphericalAngularMomentum)


def test_penalty_switches_loss_key():
    grads = _train_estimators({"system": {"lz_penalty": 1.0}})["grads"]
    assert isinstance(grads, LossAndGrad)
    assert grads.loss_key == "penalized_loss"


def test_default_loss_grads():
    grads = _train_estimators({})["grads"]
    assert isinstance(grads, LossAndGrad)
    assert grads.loss_key == "total_energy"
    assert grads.clip_method == "mad"
    assert grads.clip_scale == 100


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
    )["grads"]
    assert isinstance(grads, LossAndGrad)
    assert grads.clip_method == "none"
    np.testing.assert_allclose(grads.clip_scale, 7.0)


def test_angular_projection_is_opt_in():
    assert "angular_projection" not in _eval_estimators({})
    config = {"estimators": {"angular_projection": {"enabled": True}}}
    assert isinstance(
        _eval_estimators(config)["angular_projection"],
        SphericalAngularMomentumProjection,
    )


@pytest.mark.parametrize(
    "workflow_cls,config,match",
    [
        pytest.param(
            HallEvalWorkflow,
            {
                "system": {"lz_penalty": 1.0},
                "estimators": {
                    "energy": {"enabled": False},
                },
            },
            r"penalties require the total energy estimator",
            id="penalty-without-energy",
        ),
        *[
            pytest.param(
                HallTrainWorkflow,
                {
                    "system": {penalty: 1.0},
                    "estimators": {"angular_momentum": {"enabled": False}},
                },
                r"penalties require the angular-momentum estimator",
                id=f"{penalty}-without-angular-momentum",
            )
            for penalty in ("lz_penalty", "l2_penalty")
        ],
        pytest.param(
            HallEvalWorkflow,
            {
                "estimators": {
                    "energy": {"enabled": False},
                    "angular_projection": {"enabled": True},
                }
            },
            r"projection requires the total energy estimator",
            id="projection-without-energy",
        ),
    ],
)
def test_rejects_inconsistent_estimator_config(workflow_cls, config, match):
    with pytest.raises(ValidationError, match=match):
        workflow_cls.from_config(config)


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
