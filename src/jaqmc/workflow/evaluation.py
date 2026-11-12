# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Evaluation workflow runtime."""

import logging
import time
from collections.abc import Callable
from dataclasses import field, replace
from typing import Annotated

import jax
from jax.experimental import multihost_utils
from pydantic import Field
from upath import UPath

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.writer import Writer

from .base import Workflow, WorkflowConfig, init_batched_data
from .stage.evaluation import EvaluationWorkStage, EvaluationWorkStageConfig

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "workflow"}
)


@configurable_dataclass
class EvaluationWorkflowConfig(WorkflowConfig):
    """Workflow settings specific to evaluation.

    Args:
        source_path: Training run directory or checkpoint file used to restore
            model parameters and sampler state before evaluation.
    """

    source_path: str | None = None


@configurable_dataclass
class EvaluationWorkflow(Workflow):
    """Evaluation workflow with a runtime evaluation stage."""

    workflow_name = "evaluation"
    config: EvaluationWorkflowConfig = Field(
        default_factory=EvaluationWorkflowConfig, alias="workflow"
    )
    run_config: EvaluationWorkStageConfig = Field(
        default_factory=EvaluationWorkStageConfig, alias="run"
    )
    writers: dict[
        str, Annotated[Writer | None, ModuleConfig(package="jaqmc.writer")]
    ] = field(default_factory=dict)
    data_init: Callable[[int, PRNGKey], Data | BatchedData] = runtime_dep()
    evaluation_stage: EvaluationWorkStage = runtime_dep()

    @property
    def source_path(self) -> UPath | None:
        """Return the absolute training checkpoint location, if configured."""
        if not self.config.source_path:
            return None
        source_path = UPath(self.config.source_path)
        if not source_path.is_absolute():
            source_path = (UPath.cwd() / source_path).resolve()
        return source_path

    @property
    def source_dir(self) -> UPath | None:
        """Return the directory containing training artifacts, if configured."""
        source_path = self.source_path
        if source_path is None:
            return None
        return source_path.parent if source_path.is_file() else source_path

    def run(self) -> None:
        """Create evaluation state, restore optional training state, and run.

        Raises:
            ValueError: If trainable parameters have no source checkpoint.
        """
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
        rngs, state_rngs = jax.random.split(rngs)
        state = self.evaluation_stage.create_state(
            state_rngs, batched_data=batched_data
        )
        if (source_path := self.source_path) is not None:
            restored = self.evaluation_stage.restore_checkpoint(
                source_path,
                {
                    "params": state.params,
                    "batched_data": state.batched_data,
                    "sampler_state": state.sampler_state,
                },
                prefix="" if source_path.is_file() else "train",
                strict=True,
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
                "workflow.source_path is required."
            )
        else:
            logger.info("No training checkpoint loaded; using fresh evaluation state.")
        rngs, run_rngs = jax.random.split(rngs)
        self.evaluation_stage.run(state, context, run_rngs)
