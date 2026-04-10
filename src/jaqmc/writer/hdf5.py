# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import h5py
import jax

from jaqmc.utils.config import configurable_dataclass
from jaqmc.writer.base import Writer

__all__ = ["HDF5Writer", "h5_append"]


def h5_append(h5file: h5py.File, key: str, values: Any) -> None:
    """Append values along axis 0, creating the dataset if needed."""
    if key in h5file:
        ds = h5file[key]
        n = ds.shape[0]
        ds.resize(n + values.shape[0], axis=0)
        ds[n:] = values
    else:
        h5file.create_dataset(key, data=values, maxshape=(None, *values.shape[1:]))


@configurable_dataclass
class HDF5Writer(Writer):
    """Writes statistics to an HDF5 file.

    Existing files are truncated to ``initial_step`` data rows upon :meth:`open`
    before new rows are appended, so resumed runs discard stale rows past the
    restored checkpoint.

    Args:
        path_template: Output path template. Relative paths are resolved
            under the working directory. The template may contain ``{stage}``.
    """

    path_template: str = "{stage}_stats.h5"

    @contextmanager
    def open(self, working_dir, stage_name, initial_step: int = 0):
        save_path = self.resolve_path_template(
            working_dir, self.path_template, stage_name
        )
        open_mode = "r+b" if save_path.exists() else "w+b"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with save_path.open(open_mode) as f, h5py.File(f, "a") as self._h5_file:
            self._truncate_to(initial_step)
            yield
        self._h5_file = None

    def _truncate_to(self, initial_step: int) -> None:
        """Truncate all datasets to ``initial_step`` entries."""
        for name in list(self._h5_file):
            dataset = self._h5_file[name]
            if dataset.shape[0] > initial_step:
                dataset.resize(initial_step, axis=0)

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        if not self._h5_file:
            raise ValueError("Writing on closed file.")

        for key, value in stats.items():
            if not isinstance(value, jax.Array):
                continue
            h5_append(self._h5_file, key, value[None])
