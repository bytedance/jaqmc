# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from types import ModuleType

import h5py
import pytest
from jax import numpy as jnp

from jaqmc.writer.csv import CSVWriter
from jaqmc.writer.hdf5 import HDF5Writer
from jaqmc.writer.wandb import WandbWriter


class FakeWandbRun:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, data, step=None):
        self.logs.append((data, step))

    def finish(self):
        self.finished = True


class FakeWandb(ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.run = None
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = FakeWandbRun()
        return self.run


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


def test_wandb_writer_logs_only_scalars(tmp_path: Path, monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    writer = WandbWriter(project="project", run_name="run")

    with writer.open(tmp_path, "train"):
        writer.write(3, {"loss": 1.5, "vector": jnp.ones(2), "acceptance": 0.25})

    assert fake_wandb.init_kwargs == {
        "dir": str(tmp_path),
        "job_type": "train",
        "project": "project",
        "name": "run",
    }
    assert fake_wandb.run.logs == [({"loss": 1.5, "acceptance": 0.25}, 3)]
    assert fake_wandb.run.finished


def test_wandb_writer_reuses_active_run(tmp_path: Path, monkeypatch):
    fake_wandb = FakeWandb()
    fake_wandb.run = FakeWandbRun()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    writer = WandbWriter(project="unused")

    with writer.open(tmp_path, "train"):
        writer.write(2, {"loss": 1.0})

    assert fake_wandb.init_kwargs is None
    assert fake_wandb.run.logs == [({"loss": 1.0}, 2)]
    assert not fake_wandb.run.finished
