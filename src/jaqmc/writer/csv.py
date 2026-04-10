# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import csv
from collections.abc import Mapping
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any

from jax import numpy as jnp
from upath import UPath

from jaqmc.utils.config import configurable_dataclass
from jaqmc.writer.base import Writer

if TYPE_CHECKING:
    import _csv

__all__ = ["CSVWriter"]


@configurable_dataclass
class CSVWriter(Writer):
    """Writes statistics to a CSV file.

    Existing files are truncated to ``initial_step`` data rows upon :meth:`open`
    before new rows are appended, so resumed runs discard stale rows past the
    restored checkpoint.

    Args:
        path_template: Output path template. Relative paths are resolved
            under the working directory. The template may contain ``{stage}``.
    """

    path_template: str = "{stage}_stats.csv"

    @contextmanager
    def open(self, working_dir, stage_name, initial_step: int = 0):
        save_path = self.resolve_path_template(
            working_dir, self.path_template, stage_name
        )
        save_path.parent.mkdir(exist_ok=True, parents=True)

        self._truncate_to(save_path, initial_step)

        file_exists = False
        try:
            if save_path.exists() and save_path.stat().st_size > 0:
                file_exists = True
        except Exception:
            pass

        with save_path.open("a", newline="") as f:
            self._file: IO[str] | None = f
            self._writer: _csv._writer | None = csv.writer(f)
            self._needs_header = not file_exists
            yield

        self._file = None
        self._writer = None

    @staticmethod
    def _truncate_to(save_path: UPath, initial_step: int) -> None:
        """Keep only the header and the first ``initial_step`` data rows."""
        if not save_path.exists() or save_path.stat().st_size == 0:
            return
        with save_path.open("r", newline="") as f:
            lines = f.readlines()
        # lines[0] is the header, lines[1:] are data rows
        if len(lines) - 1 <= initial_step:
            return
        keep = lines[: 1 + initial_step]
        with save_path.open("w", newline="") as f:
            f.writelines(keep)

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Append one row of scalar statistics to the CSV file.

        Args:
            step: Current iteration step.
            stats: Statistics dictionary. Python scalars and scalar arrays are
                written; non-scalar values are skipped.

        Raises:
            ValueError: If called outside the :meth:`open` context manager.
        """
        if not self._writer:
            raise ValueError("Writing on closed file.")

        # Filter stats to keep only scalars
        scalar_stats = {}
        for k, v in stats.items():
            # Check if scalar (Python scalar) or 0-d array (JAX/NumPy)
            if jnp.isscalar(v):
                scalar_stats[k] = self.to_scalar(v)

        row_dict = {"step": step, **scalar_stats}

        # Determine columns. We assume sorted keys for consistency.
        # "step" is always first.
        columns = ["step", *sorted(scalar_stats.keys())]

        if self._needs_header:
            self._writer.writerow(columns)
            self._needs_header = False

        row = [row_dict.get(col, "") for col in columns]
        self._writer.writerow(row)

        # Ensure data is flushed to disk
        if self._file:
            self._file.flush()
