# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import jax
from jax.experimental import multihost_utils

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data

from .base import Workflow, init_batched_data
from .stage import VMCWorkStage
from .stage.base import WorkStage


def _samplers_compatible(from_stage: VMCWorkStage, to_stage: VMCWorkStage) -> bool:
    """Check if sampler state can be transferred between two stages.

    Sampler state can be transferred if both stages use the same sampler
    types for each data field.

    Args:
        from_stage: Source stage with existing sampler state.
        to_stage: Target stage that may reuse the sampler state.

    Returns:
        True if sampler states are compatible between the stages.
    """
    from_samplers = from_stage.sample_plan.samplers
    to_samplers = to_stage.sample_plan.samplers
    if from_samplers.keys() != to_samplers.keys():
        return False
    return all(type(from_samplers[k]) is type(to_samplers[k]) for k in from_samplers)


class VMCWorkflow(Workflow):
    """VMC workflow with pretrain -> train -> eval pipeline.

    Subclass and set stages in ``__init__``::

        class MyWorkflow(VMCWorkflow):
            def __init__(self, cfg):
                super().__init__(cfg)
                train = VMCWorkStage.builder(cfg.scoped("train"), wf)
                sampler = cfg.get("sampler", MCMCSampler)
                train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
                train.configure_optimizer(
                    default="jaqmc.optimizer.kfac", f_log_psi=wf.logpsi
                )
                train.configure_estimators(...)
                train.configure_loss_grads(f_log_psi=wf.logpsi)
                self.train_stage = train.build()
                self.data_init = data_init

        MyWorkflow(cfg)()

    Attributes:
        train_stage: The main training stage.
        pretrain_stage: Optional pretraining stage.
        data_init: Function to initialize electron configurations.
    """

    data_init: Callable[[int, PRNGKey], Data | BatchedData]
    pretrain_stage: VMCWorkStage | None = None
    train_stage: VMCWorkStage

    def prepare(self, dry_run: bool = False) -> None:
        super().prepare(dry_run)
        if not hasattr(self, "data_init") or not callable(self.data_init):
            raise ValueError(f"{type(self).__name__}.data_init is not configured")
        if not hasattr(self, "train_stage"):
            raise ValueError(f"{type(self).__name__}.train_stage is not configured")
        if not isinstance(self.train_stage, VMCWorkStage):
            raise ValueError(
                f"{type(self).__name__}.train_stage must be a VMCWorkStage. "
                f"Got {type(self.train_stage)}."
            )
        if self.pretrain_stage is not None and not isinstance(
            self.pretrain_stage, VMCWorkStage
        ):
            raise ValueError(
                f"{type(self).__name__}.pretrain_stage must be a VMCWorkStage. "
                f"Got {type(self.pretrain_stage)}."
            )

    def run(self) -> None:
        """Execute the pretrain -> train -> eval pipeline.

        Override in subclasses to inject pre-run logic (e.g. SCF),
        then call ``super().run(context, rngs)``.
        """
        # We must use the same seed across all processes to ensure that replicated
        # parameters are initialized identically. Generation of different keys on
        # different devices (across processes) happens later. So no fold_in here.
        rngs = jax.random.PRNGKey(self.config.seed or int(time.time()))
        rngs = multihost_utils.broadcast_one_to_all(rngs)
        context = self.run_context

        rngs, data_rngs = jax.random.split(rngs)
        batched_data = init_batched_data(
            self.data_init, self.config.batch_size, data_rngs
        )
        if self.pretrain_stage is not None:
            rngs, state_rngs, run_rngs = jax.random.split(rngs, 3)
            pretrain_state = self.pretrain_stage.create_state(
                state_rngs, batched_data=batched_data
            )
            pretrain_state = self.pretrain_stage.run(pretrain_state, context, run_rngs)
            train_inherited_state = {
                "params": pretrain_state.params,
                "batched_data": pretrain_state.batched_data,
                "sampler_state": pretrain_state.sampler_state
                if _samplers_compatible(self.pretrain_stage, self.train_stage)
                else None,
            }
        else:
            train_inherited_state = {"batched_data": batched_data}

        rngs, state_rngs, run_rngs = jax.random.split(rngs, 3)
        train_state = self.train_stage.create_state(state_rngs, **train_inherited_state)
        train_state = self.train_stage.run(train_state, context, run_rngs)

    def restore_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        stage: Literal["pretrain", "train"] = "train",
        rngs: PRNGKey | None = None,
    ):
        """Restore state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or directory.
            stage: Name of the stage to restore (``"train"`` or
                ``"pretrain"``).
            rngs: Random key for ``create_state``. Defaults to ``PRNGKey(0)``.

        Returns:
            Restored state

        Raises:
            ValueError: Invalid stage name passed.
        """
        if rngs is None:
            rngs = jax.random.PRNGKey(0)
        stage_obj: WorkStage
        if stage == "pretrain" and self.pretrain_stage is not None:
            stage_obj = self.pretrain_stage
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
