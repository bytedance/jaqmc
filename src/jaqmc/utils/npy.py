# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import io
from collections.abc import Mapping
from contextlib import contextmanager
from typing import IO, Any

import jax
import numpy as np
from numpy.lib import format as np_format
from upath import UPath

__all__ = ["NPYReadWrite"]

_MAX_ROWS_FOR_HEADER = 10**20  # Reserves 21 digits so header size never grows


def _build_padded_header(dtype: np.dtype, n_rows: int) -> bytes:
    """Serialize a ``.npy`` v2.0 header whose byte length is fixed for ``dtype``.

    Formats the header with ``shape: (_MAX_ROWS_FOR_HEADER,)`` so
    :func:`numpy.lib.format.write_array_header_2_0` sizes and 64-byte-aligns
    the header for the maximum digit width, then replaces the placeholder
    row count in-place padded with spaces.

    Returns:
        Serialized ``.npy`` v2.0 header bytes.
    """
    buf = io.BytesIO()
    np_format.write_array_header_2_0(
        buf,
        {
            "descr": np_format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": (_MAX_ROWS_FOR_HEADER,),
        },
    )
    placeholder = f"'shape': ({_MAX_ROWS_FOR_HEADER},)".encode("ascii")
    actual = f"'shape': ({n_rows},)".encode("ascii").ljust(len(placeholder), b" ")
    return buf.getvalue().replace(placeholder, actual, 1)


def _parse_header(file_obj: IO[bytes]) -> tuple[int, np.dtype, int]:
    """Read row count, structured dtype, and data offset from a ``.npy`` stream.

    Returns:
        Tuple of ``(n_rows, dtype, header_size)``.
    """
    file_obj.seek(0)
    version = np_format.read_magic(file_obj)
    read_hdr = (
        np_format.read_array_header_2_0
        if version >= (2, 0)
        else np_format.read_array_header_1_0
    )
    shape, _, dtype = read_hdr(file_obj)
    return shape[0], dtype, file_obj.tell()


class NPYReadWrite:
    """Structured ``.npy`` statistics storage.

    Stores step-level ``jax.Array`` statistics as a 1D structured NumPy
    array in a single ``.npy`` file. A fixed-size padded ``.npy`` v2.0
    header is written on the first step, each step appends one packed
    structured record, and the shape field in the header is updated in
    place and flushed so the file is always a valid ``.npy`` file between
    writes.

    Args:
        stats_path: Path of the ``.npy`` file.
    """

    def __init__(self, stats_path: UPath) -> None:
        self.stats_path = stats_path
        self._file: IO[bytes] | None = None
        self._row_buffer: np.ndarray | None = None
        self._header_size: int = 0
        self._n_rows: int = 0

    @contextmanager
    def open(self):
        """Open the ``.npy`` file for reading and appending.

        Yields:
            This ``NPYReadWrite`` instance.
        """
        self.stats_path.parent.mkdir(exist_ok=True, parents=True)
        has_existing = self.stats_path.exists() and self.stats_path.stat().st_size > 0
        with self.stats_path.open("r+b" if has_existing else "w+b") as raw:
            self._file = raw
            try:
                if has_existing:
                    self._n_rows, dtype, self._header_size = _parse_header(raw)
                    self._row_buffer = np.empty(1, dtype=dtype)
                yield self
            finally:
                raw.flush()
                self._file = None
                self._row_buffer = None
                self._header_size = 0
                self._n_rows = 0

    def truncate(self, steps: int) -> None:
        """Truncate the stored structured array to at most ``steps`` rows.

        Args:
            steps: Maximum number of rows to keep.
        """
        if not self.stats_path.exists() or self.stats_path.stat().st_size == 0:
            return
        with self.stats_path.open("r+b") as raw:
            n_rows, dtype, header_size = _parse_header(raw)
            if n_rows <= steps:
                return
            kept_rows = max(0, steps)
            raw.truncate(header_size + kept_rows * dtype.itemsize)
            raw.seek(0)
            raw.write(_build_padded_header(dtype, kept_rows))
            raw.flush()

    def read(self) -> dict[str, np.ndarray]:
        """Read all accumulated statistics from the open ``.npy`` file.

        Returns:
            Dictionary mapping stat names to NumPy arrays with a leading
            step dimension.

        Raises:
            ValueError: If called when the file is closed.
        """
        if self._file is None:
            raise ValueError("Reading from closed file.")
        if self._row_buffer is None:
            return {}
        self._file.seek(0)
        arr = np.load(self._file)
        return {name: np.asarray(arr[name]).copy() for name in (arr.dtype.names or ())}

    def write(self, stats: Mapping[str, Any]) -> None:
        """Append one step of ``jax.Array`` statistics.

        Args:
            stats: Mapping of stat names to values; only ``jax.Array``
                entries are written.

        Raises:
            ValueError: If called when the file is closed.
        """
        if self._file is None:
            raise ValueError("Writing on closed file.")

        array_stats = jax.device_get(
            {key: value for key, value in stats.items() if isinstance(value, jax.Array)}
        )
        if not array_stats:
            return

        if self._row_buffer is None:
            dtype = np.dtype(
                [
                    (key, np.asarray(val).dtype, np.asarray(val).shape)
                    for key, val in array_stats.items()
                ]
            )
            self._row_buffer = np.empty(1, dtype=dtype)
            self._header_size = len(_build_padded_header(dtype, n_rows=0))
            self._n_rows = 0

        dtype = self._row_buffer.dtype
        for name in dtype.names or ():
            self._row_buffer[name] = array_stats[name]

        self._file.seek(self._header_size + self._n_rows * dtype.itemsize)
        self._file.write(self._row_buffer.tobytes())

        self._n_rows += 1
        self._file.seek(0)
        self._file.write(_build_padded_header(dtype, self._n_rows))
        self._file.flush()
