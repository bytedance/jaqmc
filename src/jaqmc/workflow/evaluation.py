# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Evaluation workflow for MCMC sampling and observable estimation."""

import logging
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

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "workflow"}
)


@configurable_dataclass
class EvaluationWorkflowConfig(WorkflowConfig):
    """Workflow config for evaluation.

    Extends :class:`~jaqmc.workflow.base.WorkflowConfig`.

    Args:
        source_path: Training run directory or checkpoint file to load
            parameters, walkers, and sampler state from. Required when the
            initialized parameter tree has leaves; otherwise evaluation uses
            freshly initialized state.
    """

    source_path: str | None = None


class EvaluationWorkflow(Workflow):
    """Evaluation workflow for sampling and observable estimation.

    Creates fresh evaluation state, then optionally loads ``params``,
    ``batched_data``, and ``sampler_state`` from a training checkpoint when
    ``workflow.source_path`` is set. Wavefunctions with no trainable
    parameters may omit ``source_path`` and start from freshly initialized
    walkers. The evaluation stage handles its own checkpointing for
    resumability.

    Attributes:
        evaluation_stage: The evaluation stage.
        data_init: Function to initialize electron configurations.
    """

    config_class: ClassVar[type[EvaluationWorkflowConfig]] = EvaluationWorkflowConfig
    config_namespace: ClassVar[str] = "evaluation"
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
        2. Load params, data, sampler_state from training checkpoint when
           ``workflow.source_path`` is set
        3. Run evaluation (the stage handles its own checkpoint for resumability)

        Raises:
            ValueError: If ``workflow.source_path`` is omitted for a
                wavefunction with trainable parameters.
        """
        # We must use the same seed across all processes to ensure that replicated
        # parameters are initialized identically. Generation of different keys on
        # different devices (across processes) happens later. So no fold_in here.
        seed = self.config.seed if self.config.seed is not None else int(time.time())
        rngs = multihost_utils.broadcast_one_to_all(jax.random.PRNGKey(seed))
        context = self.run_context

        rngs, data_rngs = jax.random.split(rngs)
        batched_data = init_batched_data(
            self.data_init, self.config.batch_size, data_rngs
        )

        rngs, sub_rngs = jax.random.split(rngs)
        state = self.evaluation_stage.create_state(sub_rngs, batched_data=batched_data)

        source_path_str = self.config.source_path
        if source_path_str:
            source_path = UPath(source_path_str)
            if not source_path.is_absolute():
                source_path = (UPath.cwd() / source_path).resolve()

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
                source_path, wrapper, prefix=prefix, strict=True
            )
            state = replace(
                state,
                params=restored["params"],
                batched_data=restored["batched_data"],
                sampler_state=restored["sampler_state"],
            )
        elif jax.tree.leaves(state.params):
            raise ValueError(
                "This wavefunction has trainable parameters, so "
                "workflow.source_path is required. Set it to a training run "
                "directory or train_ckpt_*.npz file."
            )
        else:
            logger.info(
                "No training checkpoint loaded; starting evaluation with "
                "fresh walkers and sampler state."
            )

        rngs, sub_rngs = jax.random.split(rngs)
        self.evaluation_stage.run(state, context, sub_rngs)
