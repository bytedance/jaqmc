# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import h5py
import numpy as np
import pytest

from jaqmc.app.hydrogen_atom import (
    hydrogen_atom_eval_workflow,
    hydrogen_atom_train_workflow,
)
from jaqmc.utils.config import ConfigManager


def _train_hydrogen(save_path: Path) -> None:
    cfg = ConfigManager(
        {
            "workflow": {"save_path": str(save_path), "batch_size": 128},
            "train": {"run": {"iterations": 5}},
        }
    )
    hydrogen_atom_train_workflow(cfg)()


def _evaluation_config(
    save_path: Path,
    restore_path: Path,
    train_path: Path,
    *,
    iterations: int,
    seed: int,
    save_step_interval: int = 1000,
    save_time_interval: int = 600,
) -> ConfigManager:
    return ConfigManager(
        {
            "workflow": {
                "seed": seed,
                "save_path": str(save_path),
                "restore_path": str(restore_path),
                "source_path": str(train_path),
                "batch_size": 128,
            },
            "run": {
                "iterations": iterations,
                "save_step_interval": save_step_interval,
                "save_time_interval": save_time_interval,
            },
        }
    )


@pytest.mark.integration
def test_stats_truncated_on_restore(tmp_path: Path):
    """Stats beyond the restored checkpoint step are discarded."""
    resume_step = 3

    def create_config(
        iterations, *, seed, save_step_interval=1000, save_time_interval=600
    ):
        return ConfigManager(
            {
                "workflow": {"seed": seed, "save_path": str(tmp_path)},
                "train": {
                    "run": {
                        "iterations": iterations,
                        "save_step_interval": save_step_interval,
                        "save_time_interval": save_time_interval,
                    },
                },
            }
        )

    # Run 5 iterations with frequent checkpointing to get a mid-run checkpoint.
    # save_step_interval=3 → checkpoint at step 2 ((2+1)%3==0) and step 4 (final).
    cfg = create_config(
        iterations=5, seed=1, save_step_interval=3, save_time_interval=0
    )
    hydrogen_atom_train_workflow(cfg)()

    ckpts = sorted((tmp_path).glob("train_ckpt_*.npz"))
    ckpt_steps = [int(c.stem.split("_")[-1]) for c in ckpts]
    assert 2 in ckpt_steps, f"Expected checkpoint at step 2, got {ckpt_steps}"
    assert 4 in ckpt_steps, f"Expected checkpoint at step 4, got {ckpt_steps}"

    # Stats have 5 entries (steps 0-4).
    with h5py.File(tmp_path / "train_stats.h5", "r") as f:
        assert len(f["loss"]) == 5
        expected_hdf5_prefix = f["loss"][:resume_step].copy()
    with (tmp_path / "train_stats.csv").open(encoding="utf8") as f:
        initial_csv_lines = f.readlines()
    assert len(initial_csv_lines) == 1 + 5
    expected_csv_prefix = initial_csv_lines[: 1 + resume_step]

    # Delete the final checkpoint so restore falls back to step 2.
    (tmp_path / "train_ckpt_000004.npz").unlink()

    # Restore and continue to 8 iterations.  initial_step = 3 (checkpoint
    # step 2 + 1), so stats for steps 3-4 from the first run must be
    # discarded before appending steps 3-7.
    cfg = create_config(iterations=8, seed=2)
    hydrogen_atom_train_workflow(cfg)()

    with h5py.File(tmp_path / "train_stats.h5", "r") as f:
        np.testing.assert_array_equal(f["loss"][:resume_step], expected_hdf5_prefix)
        assert len(f["loss"]) == 8
    with (tmp_path / "train_stats.csv").open(encoding="utf8") as f:
        lines = f.readlines()
    assert lines[: 1 + resume_step] == expected_csv_prefix
    assert len(lines) == 1 + 8
    # Verify step column is monotonically increasing (no duplicates)
    steps = [int(line.split(",")[0]) for line in lines[1:]]
    assert steps == list(range(8))


@pytest.mark.integration
def test_cross_directory_resume_inherits_stats(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"

    def create_config(save_path: Path, iterations: int):
        return ConfigManager(
            {
                "workflow": {
                    "seed": 42,
                    "save_path": str(save_path),
                    "restore_path": str(source_dir)
                    if save_path == target_dir
                    else str(save_path),
                },
                "train": {
                    "run": {
                        "iterations": iterations,
                        "save_step_interval": 3,
                        "save_time_interval": 0,
                    },
                },
            }
        )

    hydrogen_atom_train_workflow(create_config(source_dir, 5))()
    hydrogen_atom_train_workflow(create_config(target_dir, 8))()

    with h5py.File(target_dir / "train_stats.h5", "r") as f:
        assert len(f["loss"]) == 8
    with (target_dir / "train_stats.csv").open(encoding="utf8") as f:
        steps = [int(line.split(",")[0]) for line in f.readlines()[1:]]
    assert steps == list(range(8))


@pytest.mark.integration
def test_evaluation_stats_truncated_on_restore(tmp_path: Path):
    """Regression test for https://github.com/bytedance/jaqmc/issues/90."""
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "evaluation"
    resume_step = 3
    _train_hydrogen(train_dir)

    cfg = _evaluation_config(
        eval_dir,
        eval_dir,
        train_dir,
        iterations=5,
        seed=1,
        save_step_interval=3,
        save_time_interval=0,
    )
    hydrogen_atom_eval_workflow(cfg)()
    with h5py.File(eval_dir / "evaluation_stats.h5", "r") as f:
        expected_prefix = f["total_energy"][:resume_step].copy()

    (eval_dir / "evaluation_ckpt_000004.npz").unlink()
    cfg = _evaluation_config(
        eval_dir,
        eval_dir,
        train_dir,
        iterations=8,
        seed=2,
    )
    hydrogen_atom_eval_workflow(cfg)()

    with h5py.File(eval_dir / "evaluation_stats.h5", "r") as f:
        np.testing.assert_array_equal(f["total_energy"][:resume_step], expected_prefix)
        assert f["total_energy"].shape[0] == 8
        expected_energy = np.nanmean(f["total_energy"], axis=0)
    with np.load(eval_dir / "evaluation_digest.npz") as digest:
        assert np.isclose(digest["total_energy"], expected_energy)


@pytest.mark.integration
def test_evaluation_cross_directory_resume_inherits_stats(tmp_path: Path):
    train_dir = tmp_path / "train"
    source_dir = tmp_path / "evaluation-source"
    target_dir = tmp_path / "evaluation-target"
    resume_step = 5
    _train_hydrogen(train_dir)

    cfg = _evaluation_config(
        source_dir,
        source_dir,
        train_dir,
        iterations=resume_step,
        seed=1,
    )
    hydrogen_atom_eval_workflow(cfg)()
    with h5py.File(source_dir / "evaluation_stats.h5", "r") as f:
        expected_prefix = f["total_energy"][:].copy()

    cfg = _evaluation_config(
        target_dir,
        source_dir,
        train_dir,
        iterations=8,
        seed=2,
    )
    hydrogen_atom_eval_workflow(cfg)()

    with h5py.File(target_dir / "evaluation_stats.h5", "r") as f:
        np.testing.assert_array_equal(f["total_energy"][:resume_step], expected_prefix)
        assert f["total_energy"].shape[0] == 8
        expected_energy = np.nanmean(f["total_energy"], axis=0)
    with np.load(target_dir / "evaluation_digest.npz") as digest:
        assert np.isclose(digest["total_energy"], expected_energy)
