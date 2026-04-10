# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Time-per-step tracker, excluding JAX JIT compilation overhead."""

import logging
import time

logger = logging.getLogger("tracker")


class TimeTracker:
    """Tracks time per step, excluding JAX JIT compilation overhead.

    The tracker intentionally excludes the first step measurement because JAX
    Just-In-Time (JIT) compilation typically occurs on the first execution of
    jitted functions, which would skew the average time per step calculation.

    Example usage in a training loop::

        def training_loop(iterations: int):
            time_tracker = TimeTracker()
            logger = logging.getLogger("train")
            for step in range(iterations):
                execute_jitted_step()
                time_tracker.tick()
            time_tracker.log_time_per_step(logger=logger)
    """

    _start_time: float | None = None
    _last_time: float | None = None
    _ticks: int = 0

    def tick(self) -> None:
        """Record a time step.

        The first call initializes the start time but is excluded from
        time-per-step calculations to account for JAX JIT compilation overhead.
        """
        current_time = time.time()
        self._start_time = self._start_time or current_time
        self._last_time = current_time
        self._ticks += 1

    def time_per_step(self) -> float | None:
        """Calculate average time per step, excluding the first (JIT) step.

        Returns:
            Average time per step in seconds, or None if fewer than 2 ticks.
        """
        if (
            self._start_time is not None
            and self._last_time is not None
            and self._ticks > 1
        ):
            return (self._last_time - self._start_time) / (self._ticks - 1)
        return None

    def log_time_per_step(
        self, logger: logging.Logger | logging.LoggerAdapter = logger
    ) -> None:
        """Log the average time per step.

        Args:
            logger: Logger instance to use.
        """
        if time_per_step := self.time_per_step():
            logger.info("Time per step: %.3fs", time_per_step)
