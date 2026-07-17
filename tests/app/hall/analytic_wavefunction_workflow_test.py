# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest

from jaqmc.app.hall import HallEvalWorkflow, HallTrainWorkflow
from jaqmc.utils.config import ConfigManager


def _hall_config(tmp_path, *, wf_module: str, save_name: str):
    return {
        "workflow": {
            "save_path": str(tmp_path / save_name),
            "batch_size": 16,
            "seed": 0,
        },
        "system": {
            "flux": 6,
            "nspins": (3, 0),
        },
        "wf": {"module": wf_module},
    }


def _evaluation_run_config():
    """Return an evaluation run configuration compatible with supported JAX."""
    return {
        "iterations": 50,
        "burn_in": 0,
        # JAX 0.5.x and 0.6.x cannot validate this Hessian path inside shard_map.
        "check_vma": jax.__version_info__[:2] not in {(0, 5), (0, 6)},
    }


@pytest.mark.parametrize("wf_module", ["laughlin", "free"])
def test_hall_train_rejects_parameter_free_wavefunction(tmp_path, wf_module):
    train_dir = tmp_path / "train"
    config = _hall_config(tmp_path, wf_module=wf_module, save_name="train")
    config["train"] = {"run": {"iterations": 1}}

    with pytest.raises(ValueError, match="no trainable parameters"):
        HallTrainWorkflow(ConfigManager(config))()

    assert not list(train_dir.glob("train_ckpt_*.npz"))


def test_laughlin_direct_eval_keeps_local_energy_variance_bounded(tmp_path):
    config = _hall_config(tmp_path, wf_module="laughlin", save_name="eval")
    config["run"] = _evaluation_run_config()
    HallEvalWorkflow(ConfigManager(config))()

    digest = np.load(tmp_path / "eval" / "evaluation_digest.npz")
    assert digest["energy:potential"] > 0
    assert digest["total_energy_real_var"] < 0.5


def test_free_direct_eval_is_non_interacting(tmp_path):
    config = _hall_config(tmp_path, wf_module="free", save_name="eval")
    config["system"]["interaction_strength"] = 0.0
    config["run"] = _evaluation_run_config()
    HallEvalWorkflow(ConfigManager(config))()

    digest = np.load(tmp_path / "eval" / "evaluation_digest.npz")
    np.testing.assert_allclose(digest["energy:potential"], 0.0, atol=1e-6)
    np.testing.assert_allclose(digest["energy:kinetic"], 1.5, atol=1e-2)
    np.testing.assert_allclose(digest["total_energy_real"], 1.5, atol=1e-2)
