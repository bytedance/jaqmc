# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared tools for generating workflow reference files."""

import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
from upath import UPath


class ReferenceJob(Protocol):
    def prepare(self, job_dir: Path) -> None:
        """Write solver input files to ``job_dir``."""

    def run(self, job_dir: Path, *, wait: bool = True) -> subprocess.Popen[bytes]:
        """Run the job prepared in ``job_dir``."""

    def finalize_npz(self, job_dir: Path, reference_path: UPath) -> Any:
        """Convert the solver output in ``job_dir`` to ``reference_path``."""


def auto_generate_reference(*, path: UPath, job: ReferenceJob) -> None:
    RUNNING, ERROR, FINISHED = 0, 1, 2
    state, error, handle = RUNNING, None, None
    temporary_directory = None
    if jax.process_index() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory = TemporaryDirectory()
        job_dir = Path(temporary_directory.name)
        job.prepare(job_dir)
        handle = job.run(job_dir, wait=False)
    while True:
        if handle is not None and (returncode := handle.poll()) is not None:
            if returncode == 0:
                try:
                    job.finalize_npz(job_dir, path)
                    state = FINISHED
                except Exception as exc:
                    error = exc
                    state = ERROR
            else:
                error = RuntimeError(f"Job exited with code {returncode}")
                state = ERROR
        state = int(multihost_utils.broadcast_one_to_all(jnp.asarray([state])).item())
        if state != RUNNING:
            break
        time.sleep(1)
    if temporary_directory is not None:
        temporary_directory.cleanup()
    if state == ERROR:
        raise error or RuntimeError("Reference generation failed on process 0.")
