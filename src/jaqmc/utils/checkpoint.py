# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile

import h5py
import jax
import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.array_types import ArrayLike

type PathLike = str | Path | UPath
"""Filesystem path accepted by checkpoint readers and writers."""

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "checkpoint"}
)


class NumPyCheckpointManager:
    """Manage saving and restoring checkpoints as NumPy ``.npz`` files.

    Checkpoints are stored as PyTrees flattened into named arrays, and can be
    restored given a reference PyTree that defines the target structure.

    Attributes:
        save_path: Base path where new checkpoints are written.
        restore_path: Path used when searching for existing checkpoints.
    """

    save_path: UPath
    restore_path: UPath

    def __init__(
        self,
        save_path: PathLike,
        restore_path: PathLike | None = None,
        *,
        prefix: str = "",
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            save_path: Base path where checkpoints will be written.
            restore_path: Optional path to search for existing checkpoints. If
                ``None``, ``save_path`` is used.
            prefix: Optional prefix for checkpoint filenames. When set,
                filenames become ``{prefix}_ckpt_{step:06d}.npz``.
        """
        self.save_path = UPath(save_path)
        self.restore_path = UPath(restore_path or save_path)
        self.prefix = f"{prefix}_" if prefix else ""

    @staticmethod
    def restore_from_file[ValueT](
        restore_path: UPath, fallback: ValueT
    ) -> tuple[int, ValueT]:
        """Restore a checkpoint from a single ``.npz`` file.

        Args:
            restore_path: Path to the checkpoint file.
            fallback: Reference PyTree used to infer the target structure and
                provide default values.

        Returns:
            A tuple ``(step, restored)``:

            - **step** -- The initial step of this run (i.e. saved step + 1).
            - **restored** -- The restored PyTree, or ``fallback`` if no valid
              checkpoint is found.

        Type Parameters:
            ValueT: Reference-tree type that is preserved in the restored value.

        Raises:
            ValueError: If ``restore_path`` is not a file.
        """
        if not restore_path.is_file():
            raise ValueError(f"{restore_path} is not a file.")
        with restore_path.open("rb") as f, np.load(f) as npf:
            logger.info("Restoring checkpoint %s", restore_path)
            step = npf["step"].item()
            data = tree_from_npz(npf, fallback)
        logger.info("Restored checkpoint %s", restore_path)
        return step + 1, cast(ValueT, data)

    def restore[ValueT](self, fallback: ValueT) -> tuple[int, ValueT]:
        """Restore the latest checkpoint from ``restore_path`` if available.

        The manager searches for the newest ``ckpt_*.npz`` file under
        ``restore_path`` (or uses it directly if it is already a file), and
        falls back to the provided reference PyTree when nothing can be
        restored.

        Args:
            fallback: Reference PyTree to use for structure and default values
                when no checkpoint exists or all are unreadable.

        Returns:
            A tuple ``(step, restored)``:

            - **step** -- The initial step of this run (i.e. saved step + 1).
            - **restored** -- The restored PyTree, or ``fallback`` if no valid
              checkpoint is found.

        Type Parameters:
            ValueT: Reference-tree type that is preserved in the restored value.
        """
        if not self.restore_path.exists():
            logger.warning("No checkpoint to restore in: %s", self.restore_path)
            return 0, fallback
        if self.restore_path.is_file():
            return self.restore_from_file(self.restore_path, fallback)
        ckpt_files = sorted(
            self.restore_path.glob(f"{self.prefix}ckpt_*.npz"), reverse=True
        )
        if not ckpt_files:
            if self.restore_path != self.save_path:
                logger.warning(
                    "Directory exists but no matching "
                    "checkpoints found: %s/%sckpt_*.npz",
                    self.restore_path,
                    self.prefix,
                )
            return 0, fallback
        for ckpt_path in ckpt_files:
            try:
                return self.restore_from_file(ckpt_path, fallback)
            except (OSError, EOFError, BadZipFile):
                logger.warning("Fail to restore checkpoint %s", ckpt_path)
        return 0, fallback

    def save(self, step: int, data):
        """Save a checkpoint for the given step.

        Args:
            step: Step index associated with this checkpoint.
            data: PyTree to serialize into the checkpoint.
        """
        # Ensure directory exists for all filesystem implementations
        self.save_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.save_path / f"{self.prefix}ckpt_{step:06d}.npz"
        logger.info("Saving checkpoint %s", ckpt_path)

        with ckpt_path.open("wb") as f:
            np.savez_compressed(f, allow_pickle=False, step=step, **tree_to_npz(data))


# https://github.com/jax-ml/jax/blob/jax-v0.8.0/jax/_src/tree_util.py#L856-L866
# Add simple entry feature for old JAX versions (<0.5.1)
def _simple_entrystr(key) -> str:
    from jax._src.lib import pytree

    match key:
        case (
            pytree.SequenceKey(idx=key)
            | pytree.DictKey(key=key)
            | pytree.GetAttrKey(name=key)
            | pytree.FlattenedIndexKey(key=key)
        ):
            return str(key)
        case _:
            return str(key)


def _pytree_key_path(key_path: jax.tree_util.KeyPath):
    """Return a stable string path for a PyTree key.

    Args:
        key_path: JAX PyTree key path for an element.

    Returns:
        Slash-separated string representation of the key path.
    """
    try:
        return jax.tree_util.keystr(key_path, simple=True, separator="/")
    except TypeError:  # No kwargs allowed
        return "/".join(map(_simple_entrystr, key_path))


def tree_to_npz(tree: Any) -> dict[str, ArrayLike]:
    """Save PyTree to npz.

    Args:
        tree: PyTree to be saved.

    Returns:
        dict of file name and arrays to be used by `np.savez`.
    """
    return {
        f"{_pytree_key_path(key_path)}": val
        for key_path, val in jax.tree_util.tree_leaves_with_path(tree)
    }


def from_npz[ValueT](
    name: str, npf: Mapping[str, np.ndarray | h5py.Group], ref_val: ValueT
) -> ValueT:
    """Restore a single value from an NPZ or HDF5 group.

    Args:
        name: Name of the stored array or dataset.
        npf: Mapping-like object representing the opened storage file.
        ref_val: Reference value whose type determines the return type and
            device placement.

    Returns:
        The restored value, matching the type of ``ref_val``.

    Type Parameters:
        ValueT: Reference value type used to select and preserve output type.
    """
    val = np.array(npf[name])
    if isinstance(ref_val, np.ndarray):
        return cast(ValueT, val)
    elif isinstance(ref_val, jnp.ndarray):
        arr = jnp.array(npf[name])
        if hasattr(ref_val, "sharding"):
            arr = jax.device_put(arr, ref_val.sharding)
        return cast(ValueT, arr)
    else:
        return cast(ValueT, val.item())


def tree_from_npz(npf: Mapping[str, np.ndarray | h5py.Group], ref_pytree: Any) -> Any:
    """Restore a PyTree from NPZ or HDF5 storage.

    Args:
        npf: Mapping-like object for the opened storage file.
        ref_pytree: Reference PyTree whose structure is used to rebuild the
            restored values.

    Returns:
        A PyTree with the same structure as ``ref_pytree`` and values loaded
        from ``npf``.
    """
    ref_vals_with_path, treedef = jax.tree_util.tree_flatten_with_path(ref_pytree)
    vals = [
        from_npz(_pytree_key_path(key_path), npf, ref_val)
        for key_path, ref_val in ref_vals_with_path
    ]
    return jax.tree.unflatten(treedef, vals)
