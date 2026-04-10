# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Evaluation workflow that loads trained params from a checkpoint."""

import time
from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar

import jax
from jax.experimental import multihost_utils
from upath import UPath

from jaqmc.utils.config import configurable_dataclass

from .base import Workflow, WorkflowConfig, init_batched_data
from .stage.evaluation import EvaluationWorkStage


@configurable_dataclass
class EvaluationWorkflowConfig(WorkflowConfig):
    """Workflow config for evaluation.

    Extends :class:`~jaqmc.workflow.base.WorkflowConfig`.

    Args:
        source_path: Path to the training run directory or checkpoint file
            to load parameters from.
    """

    source_path: str


class EvaluationWorkflow(Workflow):
    """Evaluation workflow that loads params from a training checkpoint.

    Creates fresh evaluation state (data, estimator_state), then loads
    ``params``, ``batched_data``, and ``sampler_state`` from the training
    checkpoint. The evaluation stage handles its own checkpointing for
    resumability.

    Attributes:
        eval_stage: The evaluation stage.
        data_init: Function to initialize electron configurations.
    """

    config_class: ClassVar[type[EvaluationWorkflowConfig]] = EvaluationWorkflowConfig
    config: EvaluationWorkflowConfig
    evaluation_stage: EvaluationWorkStage
    data_init: Callable

    def prepare(self, dry_run: bool = False) -> None:
        super().prepare(dry_run)
        if not hasattr(self, "data_init") or not callable(self.data_init):
            raise ValueError(f"{type(self).__name__}.data_init is not configured")
        if not hasattr(self, "evaluation_stage"):
            raise ValueError(
                f"{type(self).__name__}.evaluation_stage is not configured"
            )
        if not isinstance(self.evaluation_stage, EvaluationWorkStage):
            raise ValueError(
                f"{type(self).__name__}.evaluation_stage must be an "
                f"EvaluationWorkStage. Got {type(self.evaluation_stage)}."
            )

    def run(self) -> None:
        """Execute the evaluation workflow.

        1. Create fresh eval state (data + estimator_state as template)
        2. Load params, data, sampler_state from training checkpoint
        3. Run evaluation (the stage handles its own checkpoint for resumability)
        """
        # We must use the same seed across all processes to ensure that replicated
        # parameters are initialized identically. Generation of different keys on
        # different devices (across processes) happens later. So no fold_in here.
        rngs = jax.random.PRNGKey(self.config.seed or int(time.time()))
        rngs = multihost_utils.broadcast_one_to_all(rngs)
        context = self.run_context

        source_path = UPath(self.config.source_path)
        if not source_path.is_absolute():
            source_path = (UPath.cwd() / source_path).resolve()

        rngs, data_rngs = jax.random.split(rngs)
        batched_data = init_batched_data(
            self.data_init, self.config.batch_size, data_rngs
        )

        rngs, sub_rngs = jax.random.split(rngs)
        state = self.evaluation_stage.create_state(sub_rngs, batched_data=batched_data)

        # Load params, data, sampler_state from training checkpoint.
        # The dict wrapper matches VMCState checkpoint key paths because
        # DictKey("params") and GetAttrKey("params") both serialize to
        # "params" in the checkpoint's key path format.
        wrapper = {
            "params": state.params,
            "batched_data": state.batched_data,
            "sampler_state": state.sampler_state,
        }
        # If source_path is a file, restore directly; otherwise glob for
        # train_ckpt_*.npz in the directory.
        prefix = "" if source_path.is_file() else "train"
        restored = self.evaluation_stage.restore_checkpoint(
            source_path, wrapper, prefix=prefix
        )
        state = replace(
            state,
            params=restored["params"],
            batched_data=restored["batched_data"],
            sampler_state=restored["sampler_state"],
        )

        rngs, sub_rngs = jax.random.split(rngs)
        self.evaluation_stage.run(state, context, sub_rngs)
