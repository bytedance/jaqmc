# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from upath import UPath

from jaqmc.writer.csv import CSVWriter


def _write_rows(directory: UPath, n: int) -> None:
    writer = CSVWriter()
    with writer.open(directory, "train"):
        for step in range(n):
            writer.write(step, {"loss": float(step)})


def _data_rows(directory: UPath) -> list[str]:
    return (directory / "train_stats.csv").read_text(encoding="utf8").splitlines()[1:]


def test_same_dir_truncates_then_appends(tmp_path: Path) -> None:
    working_dir = UPath(tmp_path)
    _write_rows(working_dir, 5)
    prefix = _data_rows(working_dir)[:3]

    writer = CSVWriter()
    writer.sync_history(working_dir, working_dir, "train", 3)
    with writer.open(working_dir, "train"):
        writer.write(3, {"loss": 3.0})

    rows = _data_rows(working_dir)
    assert rows[:3] == prefix
    assert [int(row.split(",")[0]) for row in rows] == [0, 1, 2, 3]


def test_cross_dir_copies_without_mutating_source(tmp_path: Path) -> None:
    source = UPath(tmp_path) / "source"
    dest = UPath(tmp_path) / "dest"
    dest.mkdir()
    _write_rows(source, 5)
    source_rows = _data_rows(source)

    writer = CSVWriter()
    writer.sync_history(source, dest, "train", 3)
    with writer.open(dest, "train"):
        writer.write(3, {"loss": 3.0})

    assert _data_rows(source) == source_rows
    dest_rows = _data_rows(dest)
    assert dest_rows[:3] == source_rows[:3]
    assert [int(row.split(",")[0]) for row in dest_rows] == [0, 1, 2, 3]
