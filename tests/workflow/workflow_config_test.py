# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import datetime as dt

import pytest

from jaqmc.utils.config import ConfigManager
from jaqmc.workflow.base import Workflow
from jaqmc.workflow.evaluation import EvaluationWorkflowConfig
from jaqmc.workflow.stage.base import WorkStageConfig


class NamespacedWorkflow(Workflow):
    config_namespace = "custom"

    def run(self) -> None:
        raise NotImplementedError()


class EvaluationNamespacedWorkflow(NamespacedWorkflow):
    config_namespace = "evaluation"
    config_class = EvaluationWorkflowConfig

    def __init__(self, cfg: ConfigManager):
        super().__init__(cfg)
        self.run_config = cfg.get("run", WorkStageConfig)


class _FixedDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 6, 8, 12, 34, 56, tzinfo=tz)


@pytest.fixture
def freeze_backup_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import jaqmc.workflow.base as workflow_base

    monkeypatch.setattr(workflow_base.datetime, "datetime", _FixedDateTime)


def _make_workflow(
    workflow_cls: type[Workflow], tmp_path, *, seed: int, batch_size: int = 4096
) -> Workflow:
    return workflow_cls(
        ConfigManager(
            {
                "workflow": {
                    "save_path": str(tmp_path),
                    "seed": seed,
                    "batch_size": batch_size,
                }
            }
        )
    )


def test_workflow_config_path_uses_namespace():
    cfg = ConfigManager({"workflow": {"save_path": "/tmp/run"}})
    workflow = NamespacedWorkflow(cfg)
    assert str(workflow.config_path()) == "/tmp/run/custom_config.yaml"


def test_prepare_skips_identical_backup_and_suffixes_same_second_collisions(
    tmp_path, freeze_backup_time
):
    first = _make_workflow(NamespacedWorkflow, tmp_path, seed=1)
    identical = _make_workflow(NamespacedWorkflow, tmp_path, seed=1)
    second = _make_workflow(NamespacedWorkflow, tmp_path, seed=2)
    third = _make_workflow(NamespacedWorkflow, tmp_path, seed=3)

    first.prepare()
    first_inode = (tmp_path / "custom_config.yaml").stat().st_ino
    identical.prepare()
    assert not (tmp_path / "config_history").exists()

    second.prepare()
    second_inode = (tmp_path / "custom_config.yaml").stat().st_ino
    third.prepare()

    history_dir = tmp_path / "config_history"
    first_backup = history_dir / "custom_config.backup_20260608_123456.yaml"
    second_backup = history_dir / "custom_config.backup_20260608_123456_1.yaml"

    assert first_backup.exists()
    assert second_backup.exists()
    assert first_backup.read_text() == first.cfg.to_yaml()
    assert second_backup.read_text() == second.cfg.to_yaml()
    assert first_backup.stat().st_ino == first_inode
    assert second_backup.stat().st_ino == second_inode
    assert (tmp_path / "custom_config.yaml").read_text() == third.cfg.to_yaml()


def test_prepare_tracks_evaluation_config_and_consumed_run_section(
    tmp_path, freeze_backup_time
):
    first = EvaluationNamespacedWorkflow(
        ConfigManager(
            {
                "workflow": {
                    "save_path": str(tmp_path),
                    "batch_size": 128,
                    "source_path": "train-source",
                },
                "run": {"iterations": 2},
            }
        )
    )
    second = EvaluationNamespacedWorkflow(
        ConfigManager(
            {
                "workflow": {
                    "save_path": str(tmp_path),
                    "batch_size": 128,
                    "source_path": "train-source",
                },
                "run": {"iterations": 3},
            }
        )
    )

    first.prepare()
    second.prepare()

    config_path = tmp_path / "evaluation_config.yaml"
    assert config_path.exists()
    current_yaml = config_path.read_text()
    assert "source_path: train-source" in current_yaml
    assert "iterations: 3" in current_yaml

    backup_path = (
        tmp_path / "config_history" / "evaluation_config.backup_20260608_123456.yaml"
    )
    assert backup_path.exists()
    backup_yaml = backup_path.read_text()
    assert "source_path: train-source" in backup_yaml
    assert "iterations: 2" in backup_yaml
