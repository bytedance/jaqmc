# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import h5py
import jax
import numpy as np
import pytest

from jaqmc.app.hydrogen_atom import (
    hydrogen_atom_eval_workflow,
    hydrogen_atom_train_workflow,
)
from jaqmc.utils.config import ConfigManager


@pytest.mark.parametrize(
    "optimizer,lr",
    [
        ("optax:adam", 0.05),
        ("kfac", 0.5),
        pytest.param(
            "sr",
            0.5,
            marks=pytest.mark.skipif(
                jax.__version_info__ < (0, 7, 0),
                reason="jaqmc.optimizer.sr requires jax >= 0.7.0",
            ),
        ),
    ],
)
def test_simple_run(tmp_path, optimizer, lr):
    cfg = ConfigManager(
        {
            "workflow": {"save_path": str(tmp_path), "batch_size": 128},
            "train": {
                "run": {},
                "optim": {"module": optimizer, "learning_rate": {"rate": lr}},
            },
        }
    )
    hydrogen_atom_train_workflow(cfg)()

    # Check that at least one checkpoint was created in the temp directory
    ckpts = list((tmp_path).glob("train_ckpt_*.npz"))
    assert ckpts, "No checkpoints were created in the temporary directory"

    # Check that the average of the last 10 train/loss values is close to -0.5
    with h5py.File(tmp_path / "train_stats.h5", "r") as f:
        last_losses = f["loss"][-10:]
    assert np.isclose(last_losses.mean(), -0.5, atol=5e-4), (
        "Average of last 10 train/loss values is not close to -0.5"
    )


def test_evaluation_writes_per_step_stats_and_digest(tmp_path):
    train_dir = tmp_path / "train-run"
    eval_dir = tmp_path / "eval-run"

    train_cfg = ConfigManager(
        {
            "workflow": {"save_path": str(train_dir), "batch_size": 128},
            "train": {"run": {"iterations": 20}},
        }
    )
    hydrogen_atom_train_workflow(train_cfg)()

    eval_cfg = ConfigManager(
        {
            "workflow": {
                "save_path": str(eval_dir),
                "batch_size": 128,
                "source_path": str(train_dir),
            },
            "run": {"iterations": 5},
        }
    )
    hydrogen_atom_eval_workflow(eval_cfg)()

    # Per-step stats written to HDF5 with "evaluation_" prefix
    with h5py.File(eval_dir / "evaluation_stats.h5", "r") as f:
        assert "total_energy" in f
        assert f["total_energy"].shape[0] == 5

    # Digest produced after evaluation — values should be scalars
    digest_path = eval_dir / "evaluation_digest.npz"
    assert digest_path.exists()
    digest = np.load(digest_path)
    assert "total_energy" in digest
    assert digest["total_energy"].ndim == 0
    assert "energy:kinetic" in digest
    assert digest["energy:kinetic"].ndim == 0
