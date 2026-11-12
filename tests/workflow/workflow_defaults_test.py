# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from jaqmc.utils.config import configurable_dataclass
from jaqmc.workflow.base import Workflow, WorkflowConfig


@configurable_dataclass
class DefaultsWorkflowConfig(WorkflowConfig):
    value: int = 1


@configurable_dataclass
class DefaultsWorkflow(Workflow):
    config: DefaultsWorkflowConfig = Field(
        default_factory=lambda: DefaultsWorkflowConfig(seed=123), alias="workflow"
    )

    def build_runtime(self) -> None:
        pass


def test_workflow_typed_defaults_are_applied():
    workflow = DefaultsWorkflow.from_config({})

    assert workflow.config.seed == 123
    assert workflow.config.value == 1
