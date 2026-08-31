# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from typing import Any

import h5py
import jax
import numpy as np
from upath import UPath


class CachedHDF5(h5py.File):
    """HDF5 file that reuses dataset handles by name.

    JaQMC writes statistics one step at a time, but each step usually appends
    to the same small set of datasets. In h5py, ``file[name]`` and
    ``name in file`` are not ordinary dictionary operations: both ask the HDF5
    library to resolve ``name``, and ``file[name]`` also builds a new
    ``Dataset`` wrapper for the opened object. For small per-step writes, that
    repeated path lookup can dominate the cost of the actual append.

    This class caches only opened or newly created ``Dataset`` handles. It does
    not buffer rows or delay writes. Hot write paths should therefore check
    :meth:`cached_dataset` first and fall back to HDF5 lookup only for names
    that have not been seen before.
    """

    def __init__(self, name, mode="r", **kwds) -> None:
        super().__init__(name, mode, **kwds)
        self._cached_dataset_handlers: dict[str, h5py.Dataset] = {}

    def __getitem__(self, name):
        if name not in self._cached_dataset_handlers:
            self._cached_dataset_handlers[name] = super().__getitem__(name)
        return self._cached_dataset_handlers[name]

    def cached_dataset(self, name: str) -> h5py.Dataset | None:
        return self._cached_dataset_handlers.get(name)

    def cache_dataset(self, name: str, dataset: h5py.Dataset) -> None:
        self._cached_dataset_handlers[name] = dataset


class HDF5ReadWrite:
    """Statistics storage in an HDF5 file.

    Args:
        stats_path: Path of the HDF5 file.
        truncate_to: If given, all datasets are truncated to at most this
            many rows upon :meth:`open`, so resumed runs discard stale rows
            left by a previous interrupted run. If None, rows are kept.
    """

    def __init__(self, stats_path: UPath) -> None:
        self.stats_path = stats_path
        self._h5_file: CachedHDF5 | None = None

    @contextmanager
    def open(self):
        self.stats_path.parent.mkdir(exist_ok=True, parents=True)
        open_mode = "r+b" if self.stats_path.exists() else "w+b"
        with self.stats_path.open(open_mode) as raw, CachedHDF5(raw, "a") as h5:
            self._h5_file = h5
            try:
                yield h5
            finally:
                self.close()

    def truncate(self, steps: int) -> None:
        """Truncate all datasets to ``truncate_to`` rows, if set."""
        with ExitStack() as stack:
            h5_file = (
                self._h5_file
                if self._h5_file is not None
                else stack.enter_context(self.open())
            )
            for name in list(h5_file):
                dataset = h5_file[name]
                if dataset.shape[0] > steps:
                    dataset.resize(steps, axis=0)

    def read(self) -> dict[str, np.ndarray]:
        """Read all accumulated stats from the open HDF5 file.

        Returns:
            Dictionary mapping stat names to numpy arrays with a leading
            step dimension.

        Raises:
            ValueError: Writing on closed file.
        """
        if not self._h5_file:
            raise ValueError("Writing on closed file.")
        return {key: np.asarray(self._h5_file[key][:]) for key in self._h5_file}

    def write(self, stats: Mapping[str, Any]) -> None:
        if not self._h5_file:
            raise ValueError("Writing on closed file.")

        array_stats = jax.device_get(
            {key: value for key, value in stats.items() if isinstance(value, jax.Array)}
        )
        for key, value in array_stats.items():
            ds = self._h5_file.cached_dataset(key)
            if ds is None and key not in self._h5_file:
                ds = self._h5_file.create_dataset(
                    key, data=value[None], maxshape=(None, *value.shape)
                )
                self._h5_file.cache_dataset(key, ds)
            else:
                ds = ds if ds is not None else self._h5_file[key]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = value

    def close(self):
        if not self._h5_file:
            return
        self._h5_file.close()
        self._h5_file = None
