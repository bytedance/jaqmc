# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest

from jaqmc.app.hydrogen_atom import HydrogenAtomEvalWorkflow, HydrogenAtomTrainWorkflow
from jaqmc.writer.console import ConsoleWriter


@pytest.mark.parametrize(
    "optimizer,lr",
    [
        ("optax:adam", 0.05),
        ("kfac", 0.5),
        pytest.param(
            "sr",
            0.05,
            marks=pytest.mark.skipif(
                jax.__version_info__ < (0, 7, 0),
                reason="jaqmc.optimizer.sr requires jax >= 0.7.0",
            ),
        ),
    ],
)
def test_simple_run(tmp_path, optimizer, lr):
    workflow = HydrogenAtomTrainWorkflow.from_config(
        {
            "workflow": {"seed": 42, "save_path": str(tmp_path), "batch_size": 128},
            "train": {
                "run": {},
                "optim": {"module": optimizer, "learning_rate": lr},
            },
        }
    )
    assert workflow.train.writers["console"] == ConsoleWriter(
        fields="pmove:.2f,energy=total_energy:.4f,variance=total_energy_var:.4f"
    )
    workflow()

    # Check that at least one checkpoint was created in the temp directory
    ckpts = list((tmp_path).glob("train_ckpt_*.npz"))
    assert ckpts, "No checkpoints were created in the temporary directory"

    # Check that the average of the last 10 train/loss values is close to -0.5
    f = np.load(tmp_path / "train_stats.npy")
    last_losses = f["loss"][-10:]
    assert np.isclose(last_losses.mean(), -0.5, atol=1e-3), (
        "Average of last 10 train/loss values is not close to -0.5"
    )


def test_evaluation_writes_per_step_stats_and_digest(tmp_path):
    train_dir = tmp_path / "train-run"
    eval_dir = tmp_path / "eval-run"

    train_workflow = HydrogenAtomTrainWorkflow.from_config(
        {
            "workflow": {"save_path": str(train_dir), "batch_size": 128},
            "train": {"run": {"iterations": 20}},
        }
    )
    train_workflow()

    eval_workflow = HydrogenAtomEvalWorkflow.from_config(
        {
            "workflow": {
                "save_path": str(eval_dir),
                "batch_size": 128,
                "source_path": str(train_dir),
            },
            "run": {"iterations": 5},
            "writers": {
                "csv": {"module": "jaqmc.writer.csv:CSVWriter"},
            },
        }
    )
    eval_workflow()

    # Per-step stats written to NPY with "evaluation_" prefix
    f = np.load(eval_dir / "evaluation_stats.npy")
    assert "total_energy" in (f.dtype.names or ())
    assert f["total_energy"].shape[0] == 5
    expected_energy = np.nanmean(f["total_energy"], axis=0)

    with (eval_dir / "evaluation_stats.csv").open(encoding="utf8") as f:
        assert len(f.readlines()) == 6

    # Digest produced after evaluation — values should be scalars
    digest_path = eval_dir / "evaluation_digest.npz"
    assert digest_path.exists()
    digest = np.load(digest_path)
    assert "total_energy" in digest
    assert digest["total_energy"].ndim == 0
    assert np.isclose(digest["total_energy"], expected_energy)
    assert "energy:kinetic" in digest
    assert digest["energy:kinetic"].ndim == 0


def test_evaluation_requires_source_path_for_trainable_wavefunction(tmp_path):
    eval_dir = tmp_path / "eval-run"
    eval_workflow = HydrogenAtomEvalWorkflow.from_config(
        {
            "workflow": {"save_path": str(eval_dir), "batch_size": 128},
            "run": {"iterations": 1},
        }
    )

    with pytest.raises(ValueError, match=r"workflow\.source_path is required"):
        eval_workflow()

    assert not (eval_dir / "evaluation_digest.npz").exists()


def test_evaluation_fails_when_source_path_is_missing(tmp_path):
    eval_dir = tmp_path / "eval-run"
    eval_workflow = HydrogenAtomEvalWorkflow.from_config(
        {
            "workflow": {
                "save_path": str(eval_dir),
                "batch_size": 128,
                "source_path": str(tmp_path / "missing-train-run"),
            },
            "run": {"iterations": 1},
        }
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint path does not exist"):
        eval_workflow()

    assert not (eval_dir / "evaluation_digest.npz").exists()


def test_evaluation_fails_when_source_dir_has_no_train_checkpoints(tmp_path):
    source_dir = tmp_path / "empty-train-run"
    source_dir.mkdir()
    eval_dir = tmp_path / "eval-run"
    eval_workflow = HydrogenAtomEvalWorkflow.from_config(
        {
            "workflow": {
                "save_path": str(eval_dir),
                "batch_size": 128,
                "source_path": str(source_dir),
            },
            "run": {"iterations": 1},
        }
    )

    with pytest.raises(FileNotFoundError, match="No matching checkpoints found"):
        eval_workflow()

    assert not (eval_dir / "evaluation_digest.npz").exists()


def test_evaluation_accepts_checkpoint_file_source_path(tmp_path):
    train_dir = tmp_path / "train-run"
    eval_dir = tmp_path / "eval-run"

    train_workflow = HydrogenAtomTrainWorkflow.from_config(
        {
            "workflow": {"save_path": str(train_dir), "batch_size": 128},
            "train": {"run": {"iterations": 3}},
        }
    )
    train_workflow()

    checkpoint_path = max(train_dir.glob("train_ckpt_*.npz"))
    eval_workflow = HydrogenAtomEvalWorkflow.from_config(
        {
            "workflow": {
                "save_path": str(eval_dir),
                "batch_size": 128,
                "source_path": str(checkpoint_path),
            },
            "run": {"iterations": 3},
        }
    )
    eval_workflow()

    assert (eval_dir / "evaluation_digest.npz").exists()
