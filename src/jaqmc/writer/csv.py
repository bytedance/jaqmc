# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import csv
from collections.abc import Mapping
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, ClassVar

from jax import numpy as jnp
from upath import UPath

from jaqmc.utils.config import configurable_dataclass
from jaqmc.writer import Writer

if TYPE_CHECKING:
    import _csv

__all__ = ["CSVWriter"]


@configurable_dataclass
class CSVWriter(Writer):
    """Appends scalar statistics to a CSV file.

    Writes ``{stage}_stats.csv`` in the working directory. Each
    :meth:`write` call adds one row: ``step`` first, then the scalar
    entries of ``stats`` in sorted key order. Python scalars and 0-d
    arrays are written; non-scalar values are skipped.

    If the file already exists and is non-empty, rows are appended and
    the header is left unchanged. On resume, :meth:`sync_history` copies
    the file from ``source_dir`` into ``working_dir`` when those
    directories differ, then keeps only the header and the first
    ``steps`` data rows.
    """

    path_template: ClassVar[str] = "{stage}_stats.csv"

    @contextmanager
    def open(self, working_dir: UPath, stage_name: str):
        save_path = working_dir / self.path_template.format(stage=stage_name)
        save_path.parent.mkdir(exist_ok=True, parents=True)

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

    def sync_history(
        self, source_dir: UPath, working_dir: UPath, stage_name: str, steps: int
    ) -> None:
        filename = self.path_template.format(stage=stage_name)
        csv_path = source_dir / filename
        if source_dir != working_dir:
            csv_path = (
                csv_path.copy_into(working_dir)
                if csv_path.exists()
                else working_dir / filename
            )
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return
        with csv_path.open("r", newline="") as f:
            lines = f.readlines()
        # lines[0] is the header, lines[1:] are data rows
        if len(lines) - 1 <= steps:
            return
        keep = lines[: 1 + steps]
        with csv_path.open("w", newline="") as f:
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
