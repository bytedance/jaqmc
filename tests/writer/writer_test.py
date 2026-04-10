# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import h5py
import pytest

from jaqmc.writer.csv import CSVWriter
from jaqmc.writer.hdf5 import HDF5Writer


def test_csv_writer_path_template(tmp_path: Path):
    writer = CSVWriter(path_template="metrics/{stage}_stats.csv")

    with writer.open(tmp_path, "train"):
        writer.write(0, {"loss": 1.0})

    assert (tmp_path / "metrics" / "train_stats.csv").exists()


def test_hdf5_writer_path_template(tmp_path: Path):
    writer = HDF5Writer(path_template="{stage}/stats.h5")

    with writer.open(tmp_path, "evaluation"):
        writer.write(0, {})

    with h5py.File(tmp_path / "evaluation" / "stats.h5", "r"):
        pass


def test_csv_writer_rejects_unknown_template_field(tmp_path: Path):
    writer = CSVWriter(path_template="{name}_stats.csv")

    with (
        pytest.raises(ValueError, match="Only '\\{stage\\}' is supported"),
        writer.open(tmp_path, "train"),
    ):
        pass
