# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp
from upath import UPath

from jaqmc.writer import Writers
from jaqmc.writer.npy import NPYWriter


def _write_rows(directory: UPath, n: int) -> None:
    writer = NPYWriter()
    with writer.open(directory, "train"):
        for step in range(n):
            writer.write(
                step,
                {
                    "loss": jnp.asarray(float(step)),
                    "matrix": jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + step,
                    "ignored_str": "skip",
                },
            )


@pytest.mark.parametrize(
    "uri_template",
    ["{tmp_path}", "memory://jaqmc-npy-writer-test/{name}"],
)
def test_same_dir_truncates_then_appends(tmp_path: Path, uri_template: str) -> None:
    working_dir = UPath(uri_template.format(tmp_path=tmp_path, name="same_dir"))
    _write_rows(working_dir, 5)
    with (working_dir / "train_stats.npy").open("rb") as f:
        initial = np.load(f)
    prefix_loss = initial["loss"][:3].copy()
    prefix_matrix = initial["matrix"][:3].copy()

    writer = NPYWriter()
    writer.sync_history(working_dir, working_dir, "train", 3)
    with writer.open(working_dir, "train"):
        writer.write(
            3,
            {
                "loss": jnp.asarray(30.0),
                "matrix": jnp.ones((2, 3), dtype=jnp.float32),
            },
        )
        read_while_open = writer.read()

    np.testing.assert_array_equal(read_while_open["loss"][:3], prefix_loss)
    assert list(read_while_open["loss"]) == [0.0, 1.0, 2.0, 30.0]
    np.testing.assert_array_equal(read_while_open["matrix"][:3], prefix_matrix)

    with (working_dir / "train_stats.npy").open("rb") as f:
        loaded = np.load(f)
    np.testing.assert_array_equal(loaded["loss"], read_while_open["loss"])
    np.testing.assert_array_equal(loaded["matrix"], read_while_open["matrix"])


def test_cross_dir_copies_without_mutating_source(tmp_path: Path) -> None:
    source = UPath(tmp_path) / "source"
    dest = UPath(tmp_path) / "dest"
    dest.mkdir()
    _write_rows(source, 5)
    with (source / "train_stats.npy").open("rb") as f:
        source_data = np.load(f)

    writer = NPYWriter()
    writer.sync_history(source, dest, "train", 3)
    with writer.open(dest, "train"):
        writer.write(
            3,
            {
                "loss": jnp.asarray(3.0),
                "matrix": jnp.zeros((2, 3), dtype=jnp.float32),
            },
        )

    with (source / "train_stats.npy").open("rb") as f:
        source_after = np.load(f)
    np.testing.assert_array_equal(source_after["loss"], source_data["loss"])
    with (dest / "train_stats.npy").open("rb") as f:
        dest_data = np.load(f)
    np.testing.assert_array_equal(dest_data["loss"][:3], source_data["loss"][:3])
    assert list(dest_data["loss"]) == [0.0, 1.0, 2.0, 3.0]


def test_none_restore_dir_skips_history_sync(tmp_path: Path) -> None:
    writers = Writers([NPYWriter()])
    with writers.open(tmp_path, "train", restore_dir=None):
        writers.write(0, {"loss": jnp.asarray(1.0)})
    loaded = np.load(tmp_path / "train_stats.npy")
    assert list(loaded["loss"]) == [1.0]


def test_many_fields_exceeding_4kb_header(tmp_path: Path) -> None:
    """Verify header sizing scales with column count without corrupting row 0."""
    writer = NPYWriter()
    stats_0 = {
        f"very_long_metric_column_name_{i:04d}": jnp.asarray(float(i))
        for i in range(200)
    }
    stats_1 = {
        f"very_long_metric_column_name_{i:04d}": jnp.asarray(float(i + 1000))
        for i in range(200)
    }
    with writer.open(UPath(tmp_path), "train"):
        writer.write(0, stats_0)
        writer.write(1, stats_1)

    loaded = np.load(tmp_path / "train_stats.npy")
    assert loaded.shape == (2,)
    assert loaded["very_long_metric_column_name_0000"].tolist() == [0.0, 1000.0]
    assert loaded["very_long_metric_column_name_0199"].tolist() == [199.0, 1199.0]
