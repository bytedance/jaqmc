# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from upath import UPath

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Write statistics for the current step.

        Args:
            step: Current iteration step.
            stats: Dictionary of statistics to write.
        """

    def sync_history(
        self, source_dir: UPath, working_dir: UPath, stage_name: str, steps: int
    ) -> None:
        """Inherit persisted history before opening the writer.

        Called on the master process before :meth:`open`. File-based writers
        should copy their artifacts from ``source_dir`` into ``working_dir``
        when those directories differ, then discard records for step
        ``steps`` and later so a resume does not keep stale rows from an
        interrupted run. Writers that do not persist files can leave the
        default no-op.

        Args:
            source_dir: Directory that contains the checkpoint being restored.
                Same as ``working_dir`` for in-place resume.
            working_dir: Directory where this run writes outputs.
            stage_name: Name of the current stage, used to locate
                stage-specific files.
            steps: Step at which the stage will resume. Keep history for
                steps ``0 .. steps-1``.
        """
        del source_dir, working_dir, stage_name, steps

    @contextmanager
    def open(self, working_dir: UPath, stage_name: str):
        """Context manager for resource handling.

        This method manages resource lifecycle and side effects, such as
        initializing files or other I/O operations. Do file creation and
        handle opening here, not in ``__init__``.

        Args:
            working_dir: The directory where artifacts should be stored.
            stage_name: Name of the current training stage.
        """
        del working_dir, stage_name
        yield


class Writers:
    """A collection of writers with master-process guarding.

    Wraps multiple :class:`Writer` instances and ensures that
    ``sync_history``, ``open``, and ``write`` only execute on the master
    process in distributed settings.
    """

    def __init__(self, writers: Sequence[Writer] = ()):
        self._writers = list(writers)
        self._is_master = False

    def __iter__(self) -> Iterator[Writer]:
        return iter(self._writers)

    @contextmanager
    def open(
        self,
        working_dir: UPath | Path,
        stage_name: str,
        *,
        restore_dir: UPath | Path,
        is_master: bool = True,
        initial_step: int = 0,
    ):
        """Open all writers on the master process.

        Args:
            working_dir: Directory where artifacts should be stored.
            stage_name: Name of the current training stage.
            restore_dir: Directory containing history to inherit before
                opening writers. Same as ``working_dir`` for in-place resume.
            is_master: Whether this is the master process.
            initial_step: The step from which training will resume. Data
                written for steps >= ``initial_step`` should be discarded.

        Yields:
            None.
        """
        self._is_master = is_master
        working_dir = UPath(working_dir)
        restore_dir = UPath(restore_dir)
        with ExitStack() as stack:
            if self._is_master:
                for writer in self._writers:
                    writer.sync_history(
                        restore_dir, working_dir, stage_name, initial_step
                    )
                for writer in self._writers:
                    stack.enter_context(writer.open(working_dir, stage_name))
                active_writers = ", ".join(
                    type(writer).__name__ for writer in self._writers
                )
                logging.LoggerAdapter(logger, extra={"category": stage_name}).info(
                    "Active writers: %s.", active_writers or "none"
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
