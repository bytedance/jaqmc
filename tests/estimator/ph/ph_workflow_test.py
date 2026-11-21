# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import yaml
from jax import numpy as jnp
from pydantic import ValidationError

from jaqmc.app.molecule.workflow import (
    MoleculeEvalWorkflow,
    MoleculeTrainWorkflow,
)
from jaqmc.app.solid.workflow import (
    SolidEvalWorkflow,
    SolidTrainWorkflow,
)
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.ph import PHEnergy


def test_molecule_workflow_wires_ph_parallel_to_ecp():
    """Keep molecule workflows wiring PH as a runtime estimator parallel to ECP."""
    workflow = MoleculeEvalWorkflow.from_config(
        {
            "system": {
                "atoms": [
                    {"symbol": "Fe", "coords": [0.0, 0.0, 0.0]},
                    {"symbol": "Li", "coords": [2.0, 0.0, 0.0]},
                    {"symbol": "H", "coords": [0.0, 2.0, 0.0]},
                ],
                "s_z": 0,
                "pp": {"Li": "ccecp", "Fe": "ph"},
            }
        }
    )
    workflow.build_runtime()
    estimators = dict(workflow.estimators)

    assert list(estimators) == ["potential", "ecp", "ph", "total"]
    assert isinstance(estimators["ecp"], ECPEnergy)
    assert isinstance(estimators["ph"], PHEnergy)
    assert estimators["ecp"].atom_symbols == ["Fe", "Li", "H"]
    assert estimators["ph"].atom_symbols == ["Fe", "Li", "H"]
    assert estimators["ph"].ph == ["Fe"]


def test_solid_workflow_wires_ph_parallel_to_ecp(monkeypatch):
    """Solid workflows wire PH across supercells with periodic distances."""
    monkeypatch.setattr(
        "jaqmc.app.solid.workflow._load_reference",
        lambda *_: SimpleNamespace(
            get_orbital_kpoints=lambda: jnp.zeros((1, 3)),
            get_kpoint_occupancies=lambda: [],
        ),
    )
    workflow = SolidEvalWorkflow.from_config(
        {
            "system": {
                "atoms": [
                    {"symbol": "Fe", "frac_coords": [0.0, 0.0, 0.0]},
                    {"symbol": "Li", "frac_coords": [0.5, 0.5, 0.5]},
                ],
                "lattice": {"a": 8.0, "b": 8.0, "c": 8.0},
                "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
                "pp": {"Fe": "ph", "Li": "ccecp"},
                "s_z": 0.5,
            }
        }
    )
    workflow.build_runtime()
    estimators = dict(workflow.estimators)

    assert list(estimators) == ["potential", "ecp", "ph", "total"]
    assert isinstance(estimators["ecp"], ECPEnergy)
    assert isinstance(estimators["ph"], PHEnergy)
    assert estimators["ph"].atom_symbols == ["Fe", "Li", "Fe", "Li"]
    assert estimators["ph"].ph == ["Fe"]
    disp, dist = estimators["ph"].distance_fn(
        jnp.array([[0.0, 0.0, 11.0]]),
        jnp.array([[0.0, 0.0, 0.0]]),
    )
    assert jnp.allclose(disp, jnp.array([[[0.0, 0.0, -5.0]]]))
    assert jnp.allclose(dist, jnp.array([[5.0]]))


@pytest.mark.parametrize(
    ("workflow_cls", "system"),
    [
        (
            MoleculeTrainWorkflow,
            {"atoms": [{"symbol": "Fe", "coords": [0, 0, 0]}], "pp": "ph"},
        ),
        (
            SolidTrainWorkflow,
            {
                "atoms": [{"symbol": "Fe", "frac_coords": [0.0, 0.0, 0.0]}],
                "lattice": {"a": 8.0, "b": 8.0, "c": 8.0},
                "pp": "ph",
            },
        ),
    ],
)
def test_ph_workflow_rejects_kinetic_enabled_override(workflow_cls, system):
    """The system, not the config, decides whether EuclideanKinetic runs."""
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        workflow_cls.from_config(
            {"system": system},
            [
                "estimators.energy.kinetic.enabled=true",
            ],
        )


def test_molecule_workflow_normalizes_kinetic_estimator_during_parsing():
    all_electron = MoleculeTrainWorkflow.from_config(
        {"system": {"module": "atom", "symbol": "H"}}
    )
    ph = MoleculeTrainWorkflow.from_config(
        {"system": {"module": "atom", "symbol": "Fe", "pp": "ph"}}
    )

    assert list(all_electron.estimators) == ["potential", "kinetic", "total"]
    assert list(ph.estimators) == ["potential", "ph", "total"]

    energy = yaml.safe_load(ph.to_yaml())["estimators"]["energy"]
    assert "enabled" not in energy["kinetic"]
    assert "enabled" not in energy["ph"]
    assert energy["ph"]["kinetic_backend"] == "forward_laplacian"


def test_training_rejects_disabled_required_estimators():
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        MoleculeTrainWorkflow.from_config(
            {"system": {"module": "atom", "symbol": "H"}},
            ["estimators.energy.enabled=false"],
        )
