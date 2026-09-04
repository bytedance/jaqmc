# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Crash-safe checkpoints used only by long-running LIT continuations."""

import logging
import uuid
from typing import Any, cast
from zipfile import BadZipFile

import numpy as np
from upath import UPath

from jaqmc.utils.checkpoint import NumPyCheckpointManager, tree_from_npz, tree_to_npz

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra={"category": "lit"})


class LITCheckpointManager(NumPyCheckpointManager):
    """Add atomic writes and corrupt-file fallback without changing JaQMC core."""

    @staticmethod
    def restore_from_file[ValueT](
        restore_path: UPath, fallback: ValueT
    ) -> tuple[int, ValueT]:
        if not restore_path.is_file():
            raise ValueError(f"{restore_path} is not a file.")
        with (
            restore_path.open("rb") as file_obj,
            np.load(file_obj, allow_pickle=False) as archive,
        ):
            step = archive["step"].item()
            restored = tree_from_npz(archive, fallback)
        return step + 1, cast(ValueT, restored)

    def restore[ValueT](
        self, fallback: ValueT, *, strict: bool = False
    ) -> tuple[int, ValueT]:
        if not self.restore_path.exists():
            if strict:
                raise FileNotFoundError(
                    f"Checkpoint path does not exist: {self.restore_path}"
                )
            return 0, fallback
        if self.restore_path.is_file():
            return self.restore_from_file(self.restore_path, fallback)
        paths = sorted(self.restore_path.glob(f"{self.prefix}ckpt_*.npz"), reverse=True)
        if not paths:
            if strict:
                raise FileNotFoundError(
                    "No matching LIT checkpoints found in "
                    f"{self.restore_path}: {self.prefix}ckpt_*.npz"
                )
            return 0, fallback
        last_error: Exception | None = None
        for path in paths:
            try:
                return self.restore_from_file(path, fallback)
            except (
                OSError,
                EOFError,
                BadZipFile,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                logger.warning("Ignoring invalid LIT checkpoint %s", path)
        if strict:
            raise RuntimeError(
                f"Failed to restore a valid LIT checkpoint from {self.restore_path}."
            ) from last_error
        return 0, fallback

    def save(self, step: int, data) -> UPath:
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / f"{self.prefix}ckpt_{step:06d}.npz"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as file_obj:
                payload: dict[str, Any] = {"step": step}
                payload.update(tree_to_npz(data))
                np.savez_compressed(file_obj, **payload)
            temporary.rename(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path
