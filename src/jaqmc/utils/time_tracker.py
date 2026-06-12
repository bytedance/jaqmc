# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Time-per-step tracker with configurable warmup-step exclusion."""

import logging
import time

logger = logging.getLogger("tracker")


class TimeTracker:
    """Tracks time per step while excluding configurable warmup steps.

    Warmup steps are excluded for two startup effects. The exact first step may
    include JAX Just-In-Time (JIT) compilation, which would skew the average.
    Separately, JAX dispatch is asynchronous, so the next few steps can appear
    artificially fast while work is still being queued rather than waited on.
    The tracker discards a configurable number of initial steps and reports the
    steady-state average after that warmup window.

    Example usage in a training loop::

        def training_loop(iterations: int):
            time_tracker = TimeTracker(warmup_steps=10)
            logger = logging.getLogger("train")
            time_tracker.start()
            for step in range(iterations):
                execute_jitted_step()
                time_tracker.tick()
            time_tracker.log_time_per_step(logger=logger)
    """

    def __init__(self, warmup_steps: int = 10) -> None:
        """Initialize the tracker.

        Args:
            warmup_steps: Number of initial completed steps to discard before
                reporting the steady-state average time per step.

        Raises:
            ValueError: If ``warmup_steps`` is negative.
        """
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self._warmup_steps = warmup_steps
        self._boundary_time: float | None = None
        self._last_time: float | None = None
        self._ticks = 0
        self.start()

    def start(self) -> None:
        """Reset the tracker baseline to the current time."""
        self._boundary_time = time.perf_counter()
        self._last_time = None
        self._ticks = 0

    def tick(self) -> None:
        """Record a completed step.

        The tracker resets its averaging boundary after each warmup step so the
        reported average only includes measured steps.
        """
        current_time = time.perf_counter()
        self._last_time = current_time
        self._ticks += 1
        if self._ticks <= self._warmup_steps:
            self._boundary_time = current_time

    def time_per_step(self) -> float | None:
        """Calculate average time per measured step.

        Returns:
            Average time per step in seconds, or None if no measured steps have
            completed yet.
        """
        measured_steps = self._ticks - self._warmup_steps
        if (
            self._boundary_time is not None
            and self._last_time is not None
            and measured_steps > 0
        ):
            return (self._last_time - self._boundary_time) / measured_steps
        return None

    def log_time_per_step(
        self, logger: logging.Logger | logging.LoggerAdapter = logger
    ) -> None:
        """Log the average time per step.

        Args:
            logger: Logger instance to use.
        """
        time_per_step = self.time_per_step()
        if time_per_step is not None:
            logger.info("Time per step: %.3fs", time_per_step)
