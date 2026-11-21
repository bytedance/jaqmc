# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from jaqmc.app.solid.workflow import SolidEvalWorkflow, SolidTrainWorkflow


def _build_eval_with_fake_reference(tmp_path, monkeypatch, restore_dir, source_dir):
    reference = SimpleNamespace(
        get_orbital_kpoints=lambda: np.asarray([[0.25, 0.0, 0.0]]),
        get_kpoint_occupancies=lambda: [(np.asarray([0.25, 0.0, 0.0]), 1, 0)],
    )
    loaded_paths = []

    def load_reference(path, _system):
        loaded_paths.append(path)
        return reference

    monkeypatch.setattr("jaqmc.app.solid.workflow.load_reference", load_reference)
    monkeypatch.setattr(
        "jaqmc.app.solid.workflow.auto_generate_reference",
        lambda **_: pytest.fail("an existing reference should be reused"),
    )
    workflow = SolidEvalWorkflow.from_config(
        {
            "workflow": {
                "save_path": str(tmp_path / "evaluation"),
                "restore_path": str(restore_dir),
                "source_path": str(source_dir),
            },
            "wf": {"hidden_dims_single": [2], "hidden_dims_double": [2]},
        }
    )
    workflow.build_runtime()
    np.testing.assert_allclose(workflow.wf.klist, [[0.25, 0.0, 0.0]])
    return loaded_paths


def test_solid_eval_restore_reference_precedes_source_reference(tmp_path, monkeypatch):
    restore_dir = tmp_path / "restore"
    source_dir = tmp_path / "source"
    restore_dir.mkdir()
    source_dir.mkdir()
    (restore_dir / "reference.npz").touch()
    (source_dir / "reference.npz").touch()

    loaded_paths = _build_eval_with_fake_reference(
        tmp_path, monkeypatch, restore_dir, source_dir
    )

    assert loaded_paths == [restore_dir / "reference.npz"]


def test_solid_eval_reuses_source_reference_when_restore_is_missing(
    tmp_path, monkeypatch
):
    restore_dir = tmp_path / "restore"
    source_dir = tmp_path / "source"
    restore_dir.mkdir()
    source_dir.mkdir()
    (source_dir / "reference.npz").touch()

    loaded_paths = _build_eval_with_fake_reference(
        tmp_path, monkeypatch, restore_dir, source_dir
    )

    assert loaded_paths == [source_dir / "reference.npz"]


def test_solid_workflow_enables_ecp_estimator_only_when_system_has_ecp():
    all_electron = SolidTrainWorkflow.from_config(
        {"system": {"module": "rock_salt", "symbol_a": "Li", "symbol_b": "H"}}
    )
    ecp = SolidTrainWorkflow.from_config(
        {
            "system": {
                "module": "rock_salt",
                "symbol_a": "Li",
                "symbol_b": "H",
                "pp": {"Li": "ccecp"},
            }
        }
    )

    assert list(all_electron.estimators) == ["potential", "kinetic", "total"]
    assert list(ecp.estimators) == ["potential", "kinetic", "ecp", "total"]
