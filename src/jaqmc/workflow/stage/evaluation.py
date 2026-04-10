# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Evaluation work stage for sampling and observable estimation.

Accumulates per-step statistics in an HDF5 file owned by the stage and produces a
``digest.npz`` summary after all steps. Optionally logs preview digests.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import h5py
import jax
import numpy as np
from upath import UPath

from jaqmc.array_types import PRNGKey
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.writer import Writers
from jaqmc.writer.hdf5 import h5_append

from .base import RunContext, WorkStageConfig
from .sampling import SamplingStageBuilder, SamplingState, SamplingWorkStage


@configurable_dataclass
class EvaluationWorkStageConfig(WorkStageConfig):
    """Configuration for evaluation work stages.

    Args:
        digest_step_interval: Log a preview of the accumulated evaluation
            digest every this many steps. The preview shows running
            statistics (means, variances) computed from all steps so far.
            Set to 0 to only print the digest at the end.
    """

    digest_step_interval: int = 0


class HDF5ReadWrite:
    def __init__(self, working_dir: UPath, prefix: str, is_master: bool):
        self._working_dir = working_dir
        self._prefix = f"{prefix}_" if prefix else ""
        self.is_master = is_master

    @contextmanager
    def open(self):
        if self.is_master:
            self._working_dir.mkdir(exist_ok=True, parents=True)
            stats_path = self._working_dir / f"{self._prefix}stats.h5"
            open_mode = "r+b" if stats_path.exists() else "w+b"
            with stats_path.open(open_mode) as raw, h5py.File(raw, "a") as h5:
                self._stats_file = h5
                yield
        else:
            self._stats_file = None
            yield

    def read(self) -> dict[str, np.ndarray]:
        """Read all accumulated stats from the open HDF5 file.

        Returns:
            Dictionary mapping stat names to numpy arrays with a leading
            step dimension.

        Raises:
            ValueError: Writing on closed file.
        """
        if not self.is_master:
            return {}
        if not self._stats_file:
            raise ValueError("Writing on closed file.")
        return {key: np.asarray(self._stats_file[key][:]) for key in self._stats_file}

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        if not self.is_master:
            return
        if not self._stats_file:
            raise ValueError("Writing on closed file.")

        for key, value in stats.items():
            if not isinstance(value, jax.Array):
                continue
            h5_append(self._stats_file, key, value[None])


class EvalStageBuilder(SamplingStageBuilder):
    """Builder for evaluation work stages.

    Call :meth:`build` to create a fully-configured :class:`EvaluationWorkStage`.
    """

    config_class: ClassVar[type[EvaluationWorkStageConfig]] = EvaluationWorkStageConfig
    config: EvaluationWorkStageConfig

    def configure_writers(self, writers: Writers | None = None) -> None:
        """Configure writers. If no argument, loads defaults from config.

        Args:
            writers: Pre-built writers, or None to load from config.
        """
        if writers is not None:
            self.writers = writers
        else:
            writers_map = self.cfg.get_collection(
                "writers",
                defaults={},
                context={"config": self.config, "name": self.name},
            )
            self.writers = Writers(list(writers_map.values()))

    def build(self) -> "EvaluationWorkStage":
        """Build a fully-configured :class:`EvaluationWorkStage`.

        Returns:
            A fully-configured :class:`EvaluationWorkStage`.
        """
        self.ensure_configured()
        return EvaluationWorkStage(
            config=self.config,
            name=self.name,
            wavefunction=self.wavefunction,
            sample_plan=self.sample_plan,
            estimators=self.estimators,
            writers=self.writers,
        )


@dataclass(kw_only=True, eq=False)
class EvaluationWorkStage(SamplingWorkStage):
    """Evaluation work stage for sampling and observable estimation.

    Runs MCMC sampling and estimator evaluation without parameter updates.
    Writes per-step statistics through writers and produces a
    ``digest.npz`` summary after all steps complete.

    Usage::

        builder = EvaluationWorkStage.builder(cfg, wf)
        sampler = cfg.get("sampler", MCMCSampler)
        builder.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        builder.configure_estimators(kinetic=..., potential=..., total=...)
        evaluation = builder.build()
    """

    config: EvaluationWorkStageConfig
    name: str
    writers: Writers

    builder: ClassVar[type[EvalStageBuilder]] = EvalStageBuilder

    def run(self, state: Any, context: RunContext, rngs: PRNGKey) -> Any:
        save_dir, prefix, _ = self._resolve_paths(context)
        is_master = jax.process_index() == 0
        self._save_dir = save_dir
        self._prefix = prefix
        self.hdf5 = HDF5ReadWrite(save_dir, prefix, is_master)
        if self.config.digest_step_interval == 0:
            self.logger.info(
                "Evaluation digest will only be printed at the last step. Set "
                "evaluation stage config digest_step_interval to a non-zero value "
                "to enable digest previewing."
            )
        else:
            self.logger.info(
                "Evaluation digest will be logged every %d steps.",
                self.config.digest_step_interval,
            )
        with self.hdf5.open():
            state = super().run(state, context, rngs)
            if is_master:
                self.write_digest(state)
        return state

    def compute_step(
        self, state: SamplingState, rngs: PRNGKey
    ) -> tuple[dict[str, Any], SamplingState]:
        """Sample and estimate observables (no parameter update).

        Returns:
            Tuple of (per-walker local stats, updated state).
        """
        sampler_rngs, est_rngs = jax.random.split(rngs, 2)
        data, _, sampler_state = self.sample_plan.step(
            state.params, state.batched_data, state.sampler_state, sampler_rngs
        )
        step_stats, estimator_state = self.estimators.evaluate(
            state.params, data, state.estimator_state, est_rngs
        )
        return step_stats, replace(
            state,
            batched_data=data,
            sampler_state=sampler_state,
            estimator_state=estimator_state,
        )

    def write_digest(self, state: SamplingState) -> None:
        """Finalize accumulated stats and save digest."""
        if not self.estimators._reduce_keys:
            return

        stacked = self.hdf5.read()
        digest = self.estimators.digest(
            stacked, state.estimator_state, n_steps=self.config.iterations
        )
        if not digest:
            return
        prefix_str = f"{self._prefix}_" if self._prefix else ""
        digest_path = self._save_dir / f"{prefix_str}digest.npz"
        with digest_path.open("wb") as f_out:
            np.savez(f_out, **digest)
        self.logger.info("Wrote evaluation digest to %s", digest_path)

    def log_preview_digest(self, step: int, state: SamplingState) -> None:
        """Log a preview digest from the stats accumulated so far."""
        stacked = self.hdf5.read()
        if not stacked:
            return
        preview = self.estimators.finalize_stats(stacked, state.estimator_state)
        for k, v in preview.items():
            if isinstance(v, jax.Array):
                if v.size < 5 and v.ndim <= 1:
                    self.logger.info("Digest step %d for %s: %s", step, k, v)
                elif v.size < 100:
                    self.logger.info("Digest step %d for %s:\n%s", step, k, v)

    def loop(self, state: SamplingState, initial_step: int, rngs):
        """Yield ``(step, state)`` for each evaluation iteration.

        Accumulates per-step statistics in the stage-owned HDF5 file.
        """
        is_master = jax.process_index() == 0
        partition = state.partition()
        split_rngs = parallel_jax.jit_sharded(
            lambda r: tuple(jax.random.split(r)),
            in_specs=parallel_jax.DATA_PARTITION,
            out_specs=parallel_jax.DATA_PARTITION,
        )
        compute = parallel_jax.jit_sharded(
            self.compute_step,
            in_specs=(partition, parallel_jax.DATA_PARTITION),
            out_specs=(parallel_jax.SHARE_PARTITION, partition),
            check_vma=self.config.check_vma,
            donate_argnums=0,
        )
        digest_step_interval = self.config.digest_step_interval

        if initial_step == 0:
            state, rngs = self.burn_in(state, rngs)
        if initial_step >= self.config.iterations:
            return

        for step in range(initial_step, self.config.iterations):
            rngs, sub_rngs = split_rngs(rngs)
            step_stats, state = compute(state, sub_rngs)

            if is_master:
                self.hdf5.write(step, step_stats)
                self.writers.write(step, step_stats)
                if digest_step_interval and (step + 1) % digest_step_interval == 0:
                    self.log_preview_digest(step, state)
            yield step, state

        # Log at last step only if not logged.
        if is_master and (
            digest_step_interval == 0 or (step + 1) % digest_step_interval != 0
        ):
            self.log_preview_digest(step, state)
