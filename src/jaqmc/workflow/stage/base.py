# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared types and configuration for work stages."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
from upath import UPath

from jaqmc.array_types import PRNGKey
from jaqmc.utils import parallel_jax
from jaqmc.utils.checkpoint import NumPyCheckpointManager
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.signal_handler import GracefulKiller
from jaqmc.utils.time_tracker import TimeTracker
from jaqmc.writer import Writers


@dataclass
class RunContext:
    """Runtime resources shared with a work stage.

    Args:
        save_path: Directory where the stage writes checkpoints and outputs.
        restore_path: Checkpoint file or directory used to resume the stage.
        signal_handler: Handler used to detect graceful termination requests.
    """

    save_path: UPath | Path
    restore_path: UPath | Path
    signal_handler: GracefulKiller


class StageState(ABC):
    @abstractmethod
    def partition(self) -> Self:
        """Return a matching pytree of :class:`~jax.sharding.PartitionSpec`."""

    @abstractmethod
    def process_allgather(self) -> Self:
        """Gather distributed arrays from all devices to each local node."""


class StageAbort[StateT: StageState](Exception):
    """Raised by loop() to request a graceful abort (e.g. NaN detected)."""

    def __init__(self, step: int, state: StateT):
        self.step = step
        self.state = state
        super().__init__()


@configurable_dataclass
class WorkStageConfig:
    """Base configuration for work stages.

    Args:
        iterations: Total number of iterations to run.
        burn_in: Sampling iterations to discard before the main loop
            for MCMC equilibration.
        save_time_interval: Minimum wall-clock seconds between checkpoint
            saves. A checkpoint is written only when both this and
            ``save_step_interval`` are satisfied.
        save_step_interval: Save checkpoints only at steps that are
            multiples of this value.
        timing_warmup_steps: Number of initial loop steps to exclude from
            time-per-step logging.
        check_vma: Enable JAX validity checks during ``shard_map``.
    """

    # JAX 0.6.x has a bug with slogdet inside shard_map with check_vma
    check_vma: bool = jax.__version_info__[:2] != (0, 6)
    iterations: int = 100
    burn_in: int = 0
    save_time_interval: int = 10 * 60
    save_step_interval: int = 1000
    timing_warmup_steps: int = 10


class WorkStage[StateT: StageState](ABC):
    """Base class for work stages with generator-based run loop.

    Subclasses implement :meth:`loop` as a generator yielding ``(step, state)``
    tuples. :meth:`run` handles checkpoint resume/save, signal handling, writers
    lifecycle, and time-per-step logging.
    """

    config: WorkStageConfig
    name: str
    writers: Writers

    def __post_init__(self):
        self.logger = logging.LoggerAdapter(
            logging.getLogger(type(self).__module__),
            extra={"category": self.name},
        )

    def _resolve_paths(
        self, context: RunContext
    ) -> tuple[UPath, str, NumPyCheckpointManager]:
        """Resolve checkpoint save/restore paths for this stage.

        Returns:
            Tuple of (save_dir, prefix, checkpoint_manager).
        """
        save_dir = UPath(context.save_path)
        restore_path = UPath(context.restore_path)
        prefix = self.name
        return (
            save_dir,
            prefix,
            NumPyCheckpointManager(save_dir, restore_path, prefix=prefix),
        )

    def run(self, state: StateT, context: RunContext, rngs: PRNGKey) -> StateT:
        """Execute the full run loop.

        Resumes from checkpoint, opens writers, iterates the generator from
        :meth:`loop`, handles checkpoint saves and signal-based abort.

        Args:
            state: Initial sharded state.
            context: Run context with working directory and signal handler.
            rngs: Random key for the run.

        Returns:
            Final state after all iterations.

        Raises:
            SystemExit: On abort (generator-initiated or signal-initiated).
            StageAbort: Caught internally; not propagated to caller.
        """
        is_master = jax.process_index() == 0
        save_dir, prefix, ckpt = self._resolve_paths(context)

        partition = state.partition()
        initial_step, restored = ckpt.restore(state)
        state = jax.device_put(restored, parallel_jax.make_sharding(partition))
        if self.config.iterations <= initial_step:
            return state

        rngs = jax.device_put(
            jax.random.split(rngs, jax.device_count()).flatten(),
            parallel_jax.make_sharding(parallel_jax.DATA_PARTITION),
        )

        def save(step: int, st: StateT) -> None:
            gathered = st.process_allgather()
            if is_master:
                ckpt.save(step, gathered)

        remaining = self.config.iterations - initial_step
        self.logger.info("Start %s %s steps.", remaining, self.name)

        last_save_time = time.time()
        tracker = TimeTracker(warmup_steps=self.config.timing_warmup_steps)

        try:
            with self.writers.open(
                save_dir, prefix, is_master=is_master, initial_step=initial_step
            ):
                tracker.start()
                for step, state in self.loop(state, initial_step, rngs):
                    tracker.tick()

                    if context.signal_handler.exit_requested:
                        raise StageAbort(step, state)

                    now = time.time()
                    is_last = step == self.config.iterations - 1
                    time_ok = now - last_save_time > self.config.save_time_interval
                    step_ok = (step + 1) % self.config.save_step_interval == 0
                    if is_last or (time_ok and step_ok):
                        save(step, state)
                        last_save_time = now
        except StageAbort as e:
            save(e.step, e.state)
            tracker.log_time_per_step(logger=self.logger)
            raise SystemExit("=" * 30 + " ABORT " + "=" * 30) from None

        self.logger.info("Done %s %s steps.", remaining, self.name)
        tracker.log_time_per_step(logger=self.logger)
        return state

    @abstractmethod
    def loop(
        self, state: StateT, initial_step: int, rngs: PRNGKey
    ) -> Iterator[tuple[int, StateT]]:
        """Yield ``(step, state)`` tuples for each iteration.

        Subclasses must implement this generator. Raise :class:`StageAbort` to
        request a graceful abort (e.g. NaN detected).

        Args:
            state: Sharded state after checkpoint resume.
            initial_step: Step to resume from (0 if fresh).
            rngs: Sharded random keys.

        Yields:
            Tuples of ``(step_index, updated_state)``.
        """

    @abstractmethod
    def create_state(self, rngs: PRNGKey, **kwargs: Any) -> StateT:
        """Create sharded state for this stage.

        Args:
            rngs: Initial random seed for all random operations.
            **kwargs: Stage-specific arguments (e.g. ``batched_data``).

        Returns:
            Sharded state object.
        """

    def restore_checkpoint(
        self,
        checkpoint_path: str | Path | UPath,
        template: Any,
        *,
        prefix: str = "",
        strict: bool = False,
    ) -> Any:
        """Restore state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or directory.
            template: Template state for deserialization.
            prefix: Checkpoint filename prefix to match.
            strict: If ``True``, raise when no matching checkpoint can be
                restored.

        Returns:
            Restored state.
        """
        raise NotImplementedError()
