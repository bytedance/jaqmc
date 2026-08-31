# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
from upath import UPath

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.hdf5 import HDF5ReadWrite
from jaqmc.writer import Writer

__all__ = ["HDF5Writer"]


@configurable_dataclass
class HDF5Writer(Writer):
    """Appends statistics, including array-valued fields, to an HDF5 file.

    Writes ``{stage}_stats.h5`` in the working directory. Each
    :meth:`write` call appends one entry per JAX array in ``stats``
    (scalars and higher-rank arrays); other value types are skipped.
    Each key is stored as a resizable dataset whose first axis grows by
    one row per call.

    On resume, :meth:`sync_history` copies the file from ``source_dir``
    into ``working_dir`` when those directories differ, then truncates
    every dataset to ``steps`` rows. While the writer is open,
    :meth:`read` returns the accumulated datasets as NumPy arrays.
    """

    path_template: ClassVar[str] = "{stage}_stats.h5"

    def __post_init__(self) -> None:
        self._h5_writer: HDF5ReadWrite | None = None

    @contextmanager
    def open(self, working_dir: UPath, stage_name: str):
        save_path = working_dir / self.path_template.format(stage=stage_name)
        h5_writer = HDF5ReadWrite(save_path)
        self._h5_writer = h5_writer
        try:
            with h5_writer.open():
                yield
        finally:
            self._h5_writer = None

    def sync_history(
        self, source_dir: UPath, working_dir: UPath, stage_name: str, steps: int
    ) -> None:
        filename = self.path_template.format(stage=stage_name)
        h5_path = source_dir / filename
        if source_dir != working_dir:
            h5_path = (
                h5_path.copy_into(working_dir)
                if h5_path.exists()
                else working_dir / filename
            )
        if h5_path.exists():
            HDF5ReadWrite(h5_path).truncate(steps)

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Append one step of ``jax.Array`` statistics to the HDF5 file.

        Args:
            step: Current iteration step. Not stored; rows append in order.
            stats: Statistics dictionary. ``jax.Array`` values are written;
                other types are skipped.

        Raises:
            ValueError: If called outside the :meth:`open` context manager.
        """
        if not self._h5_writer:
            raise ValueError("Writing on closed file.")
        self._h5_writer.write(stats)

    def read(self) -> dict[str, np.ndarray]:
        """Return all accumulated datasets as NumPy arrays.

        Returns:
            Mapping from stat name to an array with a leading step dimension.

        Raises:
            ValueError: If called outside the :meth:`open` context manager.
        """
        if not self._h5_writer:
            raise ValueError("Reading from closed file.")
        return self._h5_writer.read()
