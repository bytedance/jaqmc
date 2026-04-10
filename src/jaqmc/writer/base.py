# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from upath import UPath


class Writer(ABC):
    """Base class for statistics writers.

    Note:
        Do all I/O setup (file creation, opening handles) in the ``open``
        context manager, not in ``__init__``. In distributed runs, multiple
        processes may share the same filesystem, and side effects in
        ``__init__`` can cause resource conflicts.
    """

    @staticmethod
    def to_scalar(val: Any) -> Any:
        """Returns a Python scalar from a JAX/NumPy scalar."""
        if hasattr(val, "item"):
            return val.item()
        return val

    @staticmethod
    def resolve_path_template(
        working_dir: UPath | Path, path_template: str, stage_name: str
    ) -> UPath:
        """Resolve a writer path template against the working directory.

        The template may contain ``{stage}``, which is replaced with the
        current stage name before the path is resolved.

        Args:
            working_dir: Base directory for relative paths.
            path_template: Absolute path or path relative to ``working_dir``.
            stage_name: Name of the current stage.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If ``stage_name`` is empty or the template uses
                unsupported fields.
        """
        if not stage_name:
            raise ValueError("File-backed writers require a non-empty stage name.")
        try:
            rendered = path_template.format(stage=stage_name)
        except KeyError as e:
            raise ValueError(
                f"Unsupported path template field '{e.args[0]}'. "
                "Only '{stage}' is supported."
            ) from None

        save_path = UPath(rendered)
        if save_path.is_absolute():
            return save_path
        return UPath(working_dir) / save_path

    @abstractmethod
    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Write statistics for the current step.

        Args:
            step: Current iteration step.
            stats: Dictionary of statistics to write.
        """

    @contextmanager
    def open(self, working_dir: UPath | Path, stage_name: str, initial_step: int = 0):
        """Context manager for resource handling.

        This method manages resource lifecycle and side effects, such as
        initializing files or other I/O operations.

        When restoring from a checkpoint, ``initial_step`` indicates where
        training will resume. Writers that persist to files should truncate
        any data beyond this point so that stale entries from a previous
        (interrupted) run are discarded.

        Args:
            working_dir: The directory where artifacts should be stored.
            stage_name: Name of the current training stage. File-backed
                writers may use it when resolving their output path template.
            initial_step: The step from which training will resume. Data
                written for steps >= ``initial_step`` should be discarded.
        """
        yield


class Writers:
    """A collection of writers with master-process guarding.

    Wraps multiple :class:`Writer` instances and ensures that ``open`` and ``write``
    only execute on the master process in distributed settings.
    """

    def __init__(self, writers: Sequence[Writer] = ()):
        self._writers = list(writers)
        self._is_master = False

    @contextmanager
    def open(
        self,
        working_dir: UPath | Path,
        stage_name: str,
        *,
        is_master: bool = True,
        initial_step: int = 0,
    ):
        """Open all writers on the master process.

        Args:
            working_dir: Directory where artifacts should be stored.
            stage_name: Name of the current training stage.
            is_master: Whether this is the master process.
            initial_step: The step from which training will resume. Data
                written for steps >= ``initial_step`` should be discarded.

        Yields:
            None.
        """
        self._is_master = is_master
        with ExitStack() as stack:
            if self._is_master:
                for writer in self._writers:
                    stack.enter_context(
                        writer.open(working_dir, stage_name, initial_step=initial_step)
                    )
            yield

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Write statistics on the master process (no-op otherwise).

        Args:
            step: Current iteration step.
            stats: Dictionary of statistics to write.
        """
        if self._is_master:
            for writer in self._writers:
                writer.write(step, stats)
