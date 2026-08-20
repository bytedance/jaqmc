# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from upath import UPath

from jaqmc.utils import reference


class FakeProcess:
    def __init__(self, poll_results: list[int | None]):
        self._poll_results = iter(poll_results)

    def poll(self) -> int | None:
        return next(self._poll_results)


class FakeJob:
    def __init__(self, process: FakeProcess):
        self.process = process
        self.prepared: list[Path] = []
        self.runs: list[tuple[Path, bool]] = []
        self.finalized: list[tuple[Path, UPath]] = []

    def prepare(self, job_dir: Path) -> None:
        self.prepared.append(job_dir)

    def run(self, job_dir: Path, *, wait: bool = True) -> subprocess.Popen[bytes]:
        self.runs.append((job_dir, wait))
        return cast(subprocess.Popen[bytes], self.process)

    def finalize_npz(self, job_dir: Path, reference_path: UPath) -> None:
        self.finalized.append((job_dir, reference_path))
        reference_path.write_bytes(b"generated")


@pytest.fixture
def process_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference.jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        reference.multihost_utils, "broadcast_one_to_all", lambda value: value
    )


def test_auto_generate_runs_job_and_writes_reference(
    tmp_path: Path, process_zero: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reference.time, "sleep", lambda _: None)
    target = UPath(tmp_path / "reference.npz")
    job = FakeJob(FakeProcess([None, 0]))

    reference.auto_generate_reference(path=target, job=job)

    assert target.read_bytes() == b"generated"
    assert job.runs == [(job.prepared[0], False)]
    assert job.finalized == [(job.prepared[0], target)]


def test_auto_generate_does_not_convert_failed_job(
    tmp_path: Path, process_zero: None
) -> None:
    job = FakeJob(FakeProcess([3]))

    with pytest.raises(RuntimeError, match="Job exited with code 3"):
        reference.auto_generate_reference(
            path=UPath(tmp_path / "reference.npz"), job=job
        )

    assert not job.finalized


def test_auto_generate_reports_process_zero_failure_to_peer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reference.jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        reference.multihost_utils,
        "broadcast_one_to_all",
        lambda _: np.asarray([1], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="Reference generation failed on process 0"):
        reference.auto_generate_reference(
            path=UPath(tmp_path / "reference.npz"),
            job=FakeJob(FakeProcess([0])),
        )
