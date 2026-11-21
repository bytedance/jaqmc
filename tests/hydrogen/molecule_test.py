# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jaqmc.app.molecule import MoleculeTrainWorkflow


def test_molecule_run(tmp_path):
    workflow = MoleculeTrainWorkflow.from_config(
        {
            "workflow": {"seed": 42, "save_path": str(tmp_path), "batch_size": 256},
            "wf": {"hidden_dims_single": [64, 64], "hidden_dims_double": [8, 8]},
            "system": {
                "atoms": [
                    {"symbol": "H", "coords": [1, 0, 0]},
                    {"symbol": "H", "coords": [-1, 0, 0]},
                ],
            },
            "pretrain": {"run": {"iterations": 100}},
            "train": {
                "run": {"burn_in": 20, "iterations": 30},
                "optim": {"learning_rate": {"rate": 0.02}},
            },
        },
    )
    workflow()

    # Check that at least one checkpoint was created in the temp directory
    ckpts = list(tmp_path.glob("train_ckpt_*.npz"))
    assert ckpts, "No checkpoints were created in the temporary directory"

    # Check that the average of the last 10 train/loss values is lower than -1
    f = np.load(tmp_path / "train_stats.npy")
    assert f["loss"][0] < 0, "Energy after pretrain is positive"
    assert f["loss"][-10:].mean() < -1, (
        "Average of last 10 train/loss values is not larger than -1"
    )


def test_without_pretrain(tmp_path):
    workflow = MoleculeTrainWorkflow.from_config(
        {
            "workflow": {"seed": 42, "save_path": str(tmp_path), "batch_size": 4},
            "wf": {"hidden_dims_single": [8, 8], "hidden_dims_double": [4, 4]},
            "system": {
                "s_z": 0.5,
                "atoms": [{"symbol": "H", "coords": [0, 0, 0]}],
            },
            "pretrain": {"run": {"iterations": 0}},
            "train": {"run": {"burn_in": 0, "iterations": 1}},
        },
    )
    workflow()

    # Smoke test the run without pretrain will work
    ckpts = list(tmp_path.glob("train_ckpt_*.npz"))
    assert ckpts, "No checkpoints were created in the temporary directory"

    # Construct a fresh workflow (without calling `restored_workflow()`) to verify
    # standalone checkpoint loading builds runtime dependencies on demand.
    restored_workflow = MoleculeTrainWorkflow.from_config(
        {
            "workflow": {"batch_size": 4},
            "wf": {"hidden_dims_single": [8, 8], "hidden_dims_double": [4, 4]},
            "system": {
                "s_z": 0.5,
                "atoms": [{"symbol": "H", "coords": [0, 0, 0]}],
            },
            "pretrain": {"run": {"iterations": 0}},
        },
    )
    restored_workflow.initialize_runtime()
    restored_state = restored_workflow.restore_checkpoint(ckpts[-1])
    with np.load(ckpts[-1]) as archive:
        np.testing.assert_allclose(
            restored_state.batched_data.data.electrons,
            archive["batched_data/data/electrons"],
        )
