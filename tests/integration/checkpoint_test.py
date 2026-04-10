# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import h5py
import pytest

from jaqmc.app.hydrogen_atom import hydrogen_atom_train_workflow
from jaqmc.utils.config import ConfigManager


@pytest.mark.integration
def test_stats_truncated_on_restore(tmp_path: Path):
    """Stats beyond the restored checkpoint step are discarded."""

    def create_config(iterations, save_step_interval=1000, save_time_interval=600):
        return ConfigManager(
            {
                "workflow": {"seed": 42, "save_path": str(tmp_path)},
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
    cfg = create_config(iterations=5, save_step_interval=3, save_time_interval=0)
    hydrogen_atom_train_workflow(cfg)()

    ckpts = sorted((tmp_path).glob("train_ckpt_*.npz"))
    ckpt_steps = [int(c.stem.split("_")[-1]) for c in ckpts]
    assert 2 in ckpt_steps, f"Expected checkpoint at step 2, got {ckpt_steps}"
    assert 4 in ckpt_steps, f"Expected checkpoint at step 4, got {ckpt_steps}"

    # Stats have 5 entries (steps 0-4).
    with h5py.File(tmp_path / "train_stats.h5", "r") as f:
        assert len(f["loss"]) == 5
    with open(tmp_path / "train_stats.csv", encoding="utf8") as f:
        assert len(f.readlines()) == 1 + 5

    # Delete the final checkpoint so restore falls back to step 2.
    (tmp_path / "train_ckpt_000004.npz").unlink()

    # Restore and continue to 8 iterations.  initial_step = 3 (checkpoint
    # step 2 + 1), so stats for steps 3-4 from the first run must be
    # discarded before appending steps 3-7.
    cfg = create_config(iterations=8)
    hydrogen_atom_train_workflow(cfg)()

    with h5py.File(tmp_path / "train_stats.h5", "r") as f:
        assert len(f["loss"]) == 8
    with open(tmp_path / "train_stats.csv", encoding="utf8") as f:
        lines = f.readlines()
        assert len(lines) == 1 + 8
        # Verify step column is monotonically increasing (no duplicates)
        steps = [int(line.split(",")[0]) for line in lines[1:]]
        assert steps == list(range(8))


@pytest.mark.integration
def test_checkpoint_restoration(tmp_path: Path):
    def create_config(iterations: int, stats_file: str):
        return ConfigManager(
            {
                "workflow": {"seed": 42, "save_path": str(tmp_path)},
                "train": {
                    "run": {"iterations": iterations},
                    "writers": {"hdf5": {"path_template": "{stage}_" + stats_file}},
                },
            }
        )

    # First training run: 5 iterations
    cfg = create_config(iterations=5, stats_file="stats1.h5")
    hydrogen_atom_train_workflow(cfg)()

    # Verify only final checkpoint (iteration 4) exists
    ckpts = list((tmp_path).glob("train_ckpt_*.npz"))
    assert ckpts, "Expected checkpoints to be created"

    assert (tmp_path / "config.yaml").exists()
    assert "iterations: 5" in (tmp_path / "config.yaml").read_text()

    ckpt_iterations = sorted(int(ckpt.stem.split("_")[-1]) for ckpt in ckpts)
    assert ckpt_iterations == [4], (
        f"Expected only checkpoint at 4, got {ckpt_iterations}"
    )

    # Verify stats have 50 iterations
    assert (tmp_path / "train_stats1.h5").exists()
    with h5py.File(tmp_path / "train_stats1.h5", "r") as f:
        assert len(f["loss"]) == 5

    assert (tmp_path / "train_stats.csv").exists()
    with open(tmp_path / "train_stats.csv", encoding="utf8") as f:
        assert len(f.readlines()) == 1 + 5

    # Second training run: restore and train 3 more iterations (total 8)
    cfg = create_config(iterations=8, stats_file="stats2.h5")
    hydrogen_atom_train_workflow(cfg)()

    # Verify only expected checkpoints exist (no extra ones)
    ckpts_after = list((tmp_path).glob("train_ckpt_*.npz"))
    ckpt_iterations_after = sorted(
        int(ckpt.stem.split("_")[-1]) for ckpt in ckpts_after
    )
    assert ckpt_iterations_after == [4, 7], (
        f"Expected checkpoints at 4 and 7, got {ckpt_iterations_after}"
    )

    assert (tmp_path / "config.yaml").exists()
    assert "iterations: 8" in (tmp_path / "config.yaml").read_text()

    # Verify restored run has 3 iterations (steps 5-7 inclusive)
    with h5py.File(tmp_path / "train_stats2.h5", "r") as f:
        assert len(f["loss"]) == 3
    with open(tmp_path / "train_stats.csv", encoding="utf8") as f:
        assert len(f.readlines()) == 1 + 8

    # Third training run: 2 more iterations, writing to the same stats file
    cfg = create_config(iterations=10, stats_file="stats2.h5")
    hydrogen_atom_train_workflow(cfg)()
    # Verify restored run has 5 iterations (steps 5-9 inclusive)
    with h5py.File(tmp_path / "train_stats2.h5", "r") as f:
        assert len(f["loss"]) == 5
    with open(tmp_path / "train_stats.csv", encoding="utf8") as f:
        assert len(f.readlines()) == 1 + 10
