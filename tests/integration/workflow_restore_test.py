# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for pretrain/train stage selection on restore.

These run real (tiny) electron-gas workflows end to end and observe stage
selection through on-disk artifacts: which checkpoints and stats files appear
in the target directory, and whether stats resume or start fresh.
"""

import h5py
import pytest

from jaqmc.app.electron_gas.workflow import ElectronGasTrainWorkflow
from jaqmc.utils.config import ConfigManager


def _config(
    save_path,
    *,
    restore_path=None,
    pretrain_iterations: int = 2,
    train_iterations: int = 3,
) -> ConfigManager:
    workflow = {"seed": 0, "save_path": str(save_path), "batch_size": 16}
    if restore_path is not None:
        workflow["restore_path"] = str(restore_path)
    return ConfigManager(
        {
            "workflow": workflow,
            "system": {"rs": 1.0, "nelectrons": 2, "s_z": 0},
            "wf": {"hidden_dims_single": [8], "hidden_dims_double": [4], "ndets": 1},
            "pretrain": {"run": {"iterations": pretrain_iterations, "burn_in": 0}},
            "train": {"run": {"iterations": train_iterations, "burn_in": 0}},
        }
    )


def _stats_length(path) -> int:
    with h5py.File(path, "r") as f:
        return len(f["loss"])


def _run_source(source) -> None:
    """Run a fresh workflow leaving pretrain (steps 0-1) and train (0-2)."""
    ElectronGasTrainWorkflow(_config(source))()
    assert list(source.glob("pretrain_ckpt_*.npz"))
    assert list(source.glob("train_ckpt_*.npz"))


@pytest.mark.integration
def test_train_checkpoint_restore_skips_pretrain(tmp_path):
    source = tmp_path / "source"
    _run_source(source)

    target = tmp_path / "target"
    ElectronGasTrainWorkflow(_config(target, restore_path=source, train_iterations=5))()

    # Pretrain never ran: no pretrain checkpoints or stats in the target.
    assert not list(target.glob("pretrain_ckpt_*.npz"))
    assert not (target / "pretrain_stats.h5").exists()
    # Train resumed from the source checkpoint (step 2) and continued to
    # iteration 5, inheriting the source stats.
    assert _stats_length(target / "train_stats.h5") == 5


@pytest.mark.integration
def test_pretrain_only_restore_resumes_pretrain_and_trains_fresh(tmp_path):
    source = tmp_path / "source"
    _run_source(source)
    # Simulate a run that only completed pretraining.
    for ckpt in source.glob("train_ckpt_*.npz"):
        ckpt.unlink()

    target = tmp_path / "target"
    ElectronGasTrainWorkflow(
        _config(target, restore_path=source, pretrain_iterations=3)
    )()

    # Pretrain ran first, resuming at step 2: it inherited the source pretrain
    # stats (2 entries) and appended one more.
    assert _stats_length(target / "pretrain_stats.h5") == 3
    assert (target / "pretrain_ckpt_000002.npz").exists()
    # Train then started fresh instead of restoring from the source.
    assert _stats_length(target / "train_stats.h5") == 3
