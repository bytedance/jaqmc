# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jax import numpy as jnp
from upath import UPath

from jaqmc.utils.config import configurable_dataclass
from jaqmc.writer.base import Writer

__all__ = ["WandbWriter"]


@configurable_dataclass
class WandbWriter(Writer):
    """Logs scalar statistics to Weights & Biases.

    The writer starts one W&B run for the JaQMC stage and sends the stage name
    as the W&B job type so training and evaluation runs can be filtered by
    stage. In advanced programmatic setups, if a W&B run is already active
    before JaQMC starts the stage, the writer reuses that run.

    Advanced W&B settings should stay in W&B's normal configuration layer:
    use ``WANDB_ENTITY`` for the account or team, ``WANDB_MODE=offline`` for
    offline logging, and ``WANDB_RUN_ID``/``WANDB_RESUME`` for resuming a run.

    Args:
        project: W&B project that contains related JaQMC runs. Use the same
            project when you want runs to appear together for comparison. If
            omitted, W&B uses its configured default or infers one.
        run_name: Optional display name for this run inside the project. If
            omitted, W&B generates a short readable name.
    """

    project: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        self._run: Any | None = None

    @contextmanager
    def open(
        self,
        working_dir: UPath | Path,
        stage_name: str,
        initial_step: int = 0,
    ):
        del initial_step
        wandb = self._import_wandb()
        active_run = getattr(wandb, "run", None)

        if active_run is not None:
            self._run = active_run
            try:
                yield
            finally:
                self._run = None
        else:
            init_kwargs = self._init_kwargs(working_dir, stage_name)
            self._run = wandb.init(**init_kwargs)
            try:
                yield
            finally:
                self._run.finish()
                self._run = None

    @staticmethod
    def _import_wandb() -> Any:
        try:
            return importlib.import_module("wandb")
        except ImportError as e:
            raise ImportError(
                "WandbWriter requires the 'wandb' package. Install it with "
                "`uv add wandb` or include it in your runtime environment."
            ) from e

    def _init_kwargs(
        self, working_dir: UPath | Path, stage_name: str
    ) -> dict[str, Any]:
        init_kwargs: dict[str, Any] = {
            "dir": str(working_dir),
            "job_type": stage_name,
        }
        optional_kwargs = {
            "project": self.project,
            "name": self.run_name,
        }
        init_kwargs.update(
            {key: value for key, value in optional_kwargs.items() if value is not None}
        )
        return init_kwargs

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        """Log scalar statistics to the active W&B run.

        Args:
            step: Current iteration step.
            stats: Statistics dictionary. Python scalars and scalar arrays are
                logged; non-scalar values are skipped.

        Raises:
            ValueError: If called outside the :meth:`open` context manager.
        """
        if not self._run:
            raise ValueError("Writing on closed W&B run.")

        scalar_stats = {}
        for key, value in stats.items():
            if jnp.isscalar(value) and jnp.isrealobj(value):
                scalar_stats[key] = self.to_scalar(value)

        if scalar_stats:
            self._run.log(scalar_stats, step=step)
