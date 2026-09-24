# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from jaqmc.app.solid.workflow import SolidEvalWorkflow
from jaqmc.workflow import EvaluationWorkflow


def test_solid_eval_restore_reference_precedes_source_reference(tmp_path, monkeypatch):
    restore_dir = tmp_path / "restore"
    source_dir = tmp_path / "source"
    restore_dir.mkdir()
    source_dir.mkdir()
    restore_path = restore_dir / "reference.npz"
    source_path = source_dir / "reference.npz"
    restore_path.touch()
    source_path.touch()
    reference = SimpleNamespace(
        get_orbital_kpoints=lambda: np.asarray([[0.25, 0.0, 0.0]]),
        get_kpoint_occupancies=lambda: [(np.asarray([0.25, 0.0, 0.0]), 1, 0)],
    )
    loaded_paths = []

    def load_reference(path, _system):
        loaded_paths.append(path)
        return reference

    monkeypatch.setattr(EvaluationWorkflow, "prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "jaqmc.app.solid.workflow.load_reference",
        load_reference,
    )
    monkeypatch.setattr(
        "jaqmc.app.solid.workflow.auto_generate_reference",
        lambda **_: pytest.fail("an existing reference should be reused"),
    )
    workflow = cast(Any, SolidEvalWorkflow.__new__(SolidEvalWorkflow))
    workflow.reference = None
    workflow.save_path = tmp_path / "evaluation"
    workflow.restore_dir = restore_dir
    workflow.source_dir = source_dir
    workflow._system_config = object()
    workflow.wf = SimpleNamespace()

    SolidEvalWorkflow.prepare(workflow)

    assert loaded_paths == [restore_path]
    assert workflow.reference is reference
    np.testing.assert_allclose(workflow.wf.klist, [[0.25, 0.0, 0.0]])


def test_solid_eval_reuses_source_reference_when_restore_is_missing(
    tmp_path, monkeypatch
):
    restore_dir = tmp_path / "restore"
    source_dir = tmp_path / "source"
    restore_dir.mkdir()
    source_dir.mkdir()
    source_path = source_dir / "reference.npz"
    source_path.touch()
    reference = SimpleNamespace(
        get_orbital_kpoints=lambda: np.asarray([[0.25, 0.0, 0.0]]),
        get_kpoint_occupancies=lambda: [(np.asarray([0.25, 0.0, 0.0]), 1, 0)],
    )
    loaded_paths = []

    def load_reference(path, _system):
        loaded_paths.append(path)
        return reference

    monkeypatch.setattr(EvaluationWorkflow, "prepare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "jaqmc.app.solid.workflow.load_reference",
        load_reference,
    )
    monkeypatch.setattr(
        "jaqmc.app.solid.workflow.auto_generate_reference",
        lambda **_: pytest.fail("the source reference should be reused"),
    )
    workflow = cast(Any, SolidEvalWorkflow.__new__(SolidEvalWorkflow))
    workflow.reference = None
    workflow.save_path = tmp_path / "evaluation"
    workflow.restore_dir = restore_dir
    workflow.source_dir = source_dir
    workflow._system_config = object()
    workflow.wf = SimpleNamespace()

    SolidEvalWorkflow.prepare(workflow)

    assert loaded_paths == [source_path]
    assert workflow.reference is reference
    np.testing.assert_allclose(workflow.wf.klist, [[0.25, 0.0, 0.0]])
