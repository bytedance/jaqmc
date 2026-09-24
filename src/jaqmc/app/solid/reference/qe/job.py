# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Quantum ESPRESSO reference job execution."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from upath import UPath

from jaqmc.app.solid.config import SolidConfig, SolidQESolverConfig

from .convert import convert_save_directory
from .input import render_input


@dataclass(frozen=True)
class SolidQEReferenceJob:
    """Quantum ESPRESSO reference job for a system and solver configuration."""

    system: SolidConfig
    solver: SolidQESolverConfig

    def prepare(self, job_dir: Path) -> None:
        """Validate the configuration and write ``qe.in`` to ``job_dir``."""
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "qe.in").write_text(
            render_input(self.system, self.solver),
            encoding="utf-8",
        )

    @property
    def command(self) -> list[str]:
        return ["pw.x", "-in", "qe.in"]

    @property
    def output_path(self) -> Path:
        return Path(f"{self.solver.prefix}.save")

    def run(self, job_dir: Path, *, wait: bool = True) -> subprocess.Popen[bytes]:
        """Run the prepared QE job.

        Returns:
            The solver process handle.

        Raises:
            RuntimeError: If ``pw.x`` is missing from ``PATH``, or a
                synchronous run exits unsuccessfully.
        """
        try:
            process = subprocess.Popen(self.command, cwd=job_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Quantum ESPRESSO executable 'pw.x' was not found on PATH."
            ) from exc
        if wait and (returncode := process.wait()):
            raise RuntimeError(
                f"Quantum ESPRESSO job failed with exit status {returncode}."
            )
        return process

    def finalize_npz(self, job_dir: Path, reference_path: UPath) -> None:
        """Convert the QE save directory to the requested reference file."""
        convert_save_directory(job_dir / self.output_path, reference_path)
