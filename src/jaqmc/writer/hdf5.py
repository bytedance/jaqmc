# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from upath import UPath

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.hdf5 import HDF5ReadWrite
from jaqmc.writer.base import Writer

__all__ = ["HDF5Writer"]


@configurable_dataclass
class HDF5Writer(Writer):
    """Writes statistics to an HDF5 file.

    Existing files are truncated to ``initial_step`` data rows upon :meth:`open`
    before new rows are written, so resumed runs discard stale rows past the
    restored checkpoint.

    Args:
        path_template: Output path template. Relative paths are resolved
            under the working directory. The template may contain ``{stage}``.
    """

    path_template: str = "{stage}_stats.h5"

    @contextmanager
    def open(self, working_dir: UPath | Path, stage_name: str, initial_step: int = 0):
        save_path = self.resolve_path_template(
            working_dir, self.path_template, stage_name
        )
        self._h5_writer = HDF5ReadWrite(save_path, truncate_to=initial_step)
        with self._h5_writer.open():
            yield

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        if not self._h5_writer:
            raise ValueError("Writing on closed file.")
        self._h5_writer.write(stats)
