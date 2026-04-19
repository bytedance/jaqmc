# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from jaqmc.utils.config import ConfigManager
from jaqmc.workflow.base import Workflow


class PresetWorkflow(Workflow):
    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        return {"workflow": {"seed": 123}, "value": 1}

    def run(self) -> None:
        raise NotImplementedError()


def test_workflow_default_preset_applied():
    cfg = ConfigManager({})
    workflow = PresetWorkflow(cfg)
    assert workflow.config.seed == 123
    assert cfg.get("value", 0) == 1
    cfg.finalize()


def test_workflow_default_preset_is_low_priority():
    cfg = ConfigManager({"workflow": {"seed": 11}})
    workflow = PresetWorkflow(cfg)
    assert workflow.config.seed == 11
    cfg.finalize()
