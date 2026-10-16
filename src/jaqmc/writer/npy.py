# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
from upath import UPath

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.npy import NPYReadWrite
from jaqmc.writer import Writer

__all__ = ["NPYWriter"]


@configurable_dataclass
class NPYWriter(Writer):
    """Appends statistics, including array-valued fields, to a structured ``.npy`` file.

    Writes ``{stage}_stats.npy`` in the working directory as a 1D
    structured NumPy array readable via :func:`numpy.load`. Each
    :meth:`write` call appends one record containing all ``jax.Array``
    entries in ``stats`` (scalars and higher-rank arrays); other value
    types are skipped.

    On resume, :meth:`sync_history` copies the file from ``source_dir``
    into ``working_dir`` when those directories differ, then truncates
    the structured array to ``steps`` rows. While the writer is open,
    :meth:`read` returns the accumulated fields as a dictionary of
    NumPy arrays with a leading step dimension.
    """

    path_template: ClassVar[str] = "{stage}_stats.npy"

    def __post_init__(self) -> None:
        self._npy_writer: NPYReadWrite | None = None

    @contextmanager
    def open(self, working_dir: UPath, stage_name: str):
        save_path = working_dir / self.path_template.format(stage=stage_name)
        npy_writer = NPYReadWrite(save_path)
        self._npy_writer = npy_writer
        try:
            with npy_writer.open():
                yield
        finally:
            self._npy_writer = None

    def sync_history(
        self, source_dir: UPath, working_dir: UPath, stage_name: str, steps: int
    ) -> None:
        filename = self.path_template.format(stage=stage_name)
        npy_path = source_dir / filename
        if source_dir != working_dir:
            npy_path = (
                npy_path.copy_into(working_dir)
                if npy_path.exists()
                else working_dir / filename
            )
        if npy_path.exists():
            NPYReadWrite(npy_path).truncate(steps)

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Append one step of ``jax.Array`` statistics to the ``.npy`` file.

        Args:
            step: Current iteration step. Not stored; rows append in order.
            stats: Statistics dictionary. ``jax.Array`` values are written;
                other types are skipped.

        Raises:
            ValueError: If called outside the :meth:`open` context manager.
        """
        del step
        if not self._npy_writer:
            raise ValueError("Writing on closed file.")
        self._npy_writer.write(stats)

    def read(self) -> dict[str, np.ndarray]:
        """Return all accumulated fields as NumPy arrays.

        Returns:
            Mapping from stat name to an array with a leading step dimension.

        Raises:
            ValueError: If called outside the :meth:`open` context manager.
        """
        if not self._npy_writer:
            raise ValueError("Reading from closed file.")
        return self._npy_writer.read()
