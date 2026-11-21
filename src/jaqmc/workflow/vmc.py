# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Common VMC workflow runtime."""

import time
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import Annotated, Literal

import jax
from jax.experimental import multihost_utils

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.optimizer.base import OptimizerLike
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.optax import adam
from jaqmc.optimizer.schedule import Constant
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import UNSET, check_wired, runtime_dep
from jaqmc.writer import Writer
from jaqmc.writer.console import ConsoleWriter
from jaqmc.writer.csv import CSVWriter
from jaqmc.writer.npy import NPYWriter

from .base import Workflow, init_batched_data
from .stage.base import WorkStage
from .stage.vmc import VMCWorkStage, VMCWorkStageConfig

type WriterMap = dict[
    str, Annotated[Writer | None, ModuleConfig(package="jaqmc.writer")]
]


def stage_writers(console_fields: str | None = None) -> WriterMap:
    """Return the default console, NPY, and CSV writers for a stage.

    Args:
        console_fields: Console field spec. ``None`` keeps the console default.

    Returns:
        Writer instances keyed by ``console``, ``npy``, and ``csv``.
    """
    console = (
        ConsoleWriter()
        if console_fields is None
        else ConsoleWriter(fields=console_fields)
    )
    return {"console": console, "npy": NPYWriter(), "csv": CSVWriter()}


@configurable_dataclass
class VMCTrainConfig:
    """Configuration for a VMC training stage."""

    run: VMCWorkStageConfig = field(default_factory=VMCWorkStageConfig)
    optim: Annotated[OptimizerLike, ModuleConfig(package="jaqmc.optimizer")] = field(
        default_factory=KFACOptimizer
    )
    grads: Annotated[LossAndGrad, ModuleConfig(package="jaqmc.estimator")] = field(
        default_factory=LossAndGrad
    )
    writers: WriterMap = field(default_factory=stage_writers)


@configurable_dataclass
class VMCPretrainConfig:
    """Configuration for a VMC pretraining stage."""

    run: VMCWorkStageConfig = field(default_factory=VMCWorkStageConfig)
    optim: Annotated[OptimizerLike, ModuleConfig(package="jaqmc.optimizer")] = field(
        default_factory=lambda: adam(learning_rate=Constant(rate=3e-4))
    )
    writers: WriterMap = field(default_factory=stage_writers)


def _samplers_compatible(from_stage: VMCWorkStage, to_stage: VMCWorkStage) -> bool:
    """Return whether sampler state can be transferred between stages."""
    from_samplers = from_stage.sample_plan.samplers
    to_samplers = to_stage.sample_plan.samplers
    return from_samplers.keys() == to_samplers.keys() and all(
        type(from_samplers[name]) is type(to_samplers[name]) for name in from_samplers
    )


@configurable_dataclass
class VMCWorkflow(Workflow):
    """VMC workflow with optional pretraining and a training stage."""

    workflow_name = "train"
    data_init: Callable[[int, PRNGKey], Data | BatchedData] = runtime_dep()
    pretrain_stage: VMCWorkStage | None = runtime_dep(default=None)
    train_stage: VMCWorkStage = runtime_dep()

    def run(self) -> None:
        """Execute the pretrain then training pipeline."""
        # We must use the same seed across all processes to ensure that replicated
        # parameters are initialized identically. Generation of different keys on
        # different devices (across processes) happens later. So no fold_in here.
        seed = int(
            multihost_utils.broadcast_one_to_all(
                self.config.seed if self.config.seed is not None else int(time.time())
            )
        )
        rngs = jax.random.key(seed)
        context = self._make_run_context()

        rngs, data_rngs = jax.random.split(rngs)
        batched_data = init_batched_data(
            self.data_init, self.config.batch_size, data_rngs
        )
        stage_to_run_first = self.stage_to_run_first()
        if self.pretrain_stage is not None and stage_to_run_first == "pretrain":
            rngs, state_rngs, run_rngs = jax.random.split(rngs, 3)
            pretrain_state = self.pretrain_stage.create_state(
                state_rngs, batched_data=batched_data
            )
            pretrain_state = self.pretrain_stage.run(pretrain_state, context, run_rngs)
            inherited_state = {
                "params": pretrain_state.params,
                "batched_data": pretrain_state.batched_data,
                "sampler_state": (
                    pretrain_state.sampler_state
                    if _samplers_compatible(self.pretrain_stage, self.train_stage)
                    else None
                ),
            }
            # Train stage should not attempt to restore any more after we done pretrain.
            context.restore_path = None
        else:
            inherited_state = {"batched_data": batched_data}
        rngs, state_rngs, run_rngs = jax.random.split(rngs, 3)
        train_state = self.train_stage.create_state(state_rngs, **inherited_state)
        self.train_stage.run(train_state, context, run_rngs)

    def stage_to_run_first(self) -> Literal["train", "pretrain"]:
        restore_path = self.restore_path
        if restore_path.is_file():
            return "train" if restore_path.name.startswith("train") else "pretrain"
        return "train" if any(restore_path.glob("train_ckpt_*.npz")) else "pretrain"

    def restore_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        stage: Literal["pretrain", "train"] = "train",
        rngs: PRNGKey | None = None,
    ):
        """Restore state for a configured VMC stage.

        Builds the workflow runtime first if it has not been initialized yet.

        Returns:
            The restored stage state.

        Raises:
            ValueError: If the requested stage is not configured.
        """
        if self.train_stage is UNSET or self.data_init is UNSET:
            self.build_runtime()
        check_wired(self)
        rngs = jax.random.key(0) if rngs is None else rngs
        if stage == "pretrain" and self.pretrain_stage is not None:
            stage_obj: WorkStage = self.pretrain_stage
        elif stage == "train":
            stage_obj = self.train_stage
        else:
            raise ValueError(f"Invalid stage name {stage}.")
        data_rngs, state_rngs = jax.random.split(rngs)
        batched_data = init_batched_data(
            self.data_init, self.config.batch_size, data_rngs
        )
        template = stage_obj.create_state(state_rngs, batched_data=batched_data)
        return stage_obj.restore_checkpoint(checkpoint_path, template, prefix=stage)
