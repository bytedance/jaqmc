# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from jax import numpy as jnp
from pydantic import ValidationError

from jaqmc.app.moire.data import data_init
from jaqmc.app.moire.workflow import (
    MoireEvalWorkflow,
    MoireJointMCMCSampler,
    MoireTrainWorkflow,
)


def _base_config(tmp_path: Path) -> dict:
    return {
        "workflow": {"batch_size": 4, "save_path": str(tmp_path / "run")},
        "system": {
            "electron_spins": [4, 0],
            "nelec": 4,
            "supercell_matrix": [[2, 0], [0, 2]],
        },
        "wf": {"hidden_dims_single": [8], "hidden_dims_double": [4], "ndets": 1},
    }


def test_saved_config_excludes_runtime_klist_arrays(tmp_path: Path):
    cfg = _base_config(tmp_path)
    workflow = MoireTrainWorkflow.from_config(cfg)
    yaml = workflow.to_yaml()

    assert "singlephase_klists" not in yaml
    assert "multiphase_klist" not in yaml
    assert "translation_vectors" not in yaml


def test_runtime_owned_energy_fields_are_wired_not_serialized(tmp_path: Path):
    workflow = MoireTrainWorkflow.from_config(_base_config(tmp_path))
    energy_config = yaml.safe_load(workflow.to_yaml())["estimators"]["energy"]

    assert "prefactor" not in energy_config["kinetic"]

    workflow.build_runtime()

    expected_prefactor = (
        workflow.system.kinetic_prefactor_mev / workflow.system.effective_mass
    )
    energy = workflow.estimators.energy
    assert energy.kinetic.prefactor == pytest.approx(expected_prefactor)
    assert energy.soc.prefactor == pytest.approx(expected_prefactor)


def test_total_energy_sums_only_enabled_energy_components(tmp_path: Path):
    cfg = _base_config(tmp_path)
    cfg["wf"]["nlayers"] = 3
    cfg["estimators"] = {
        "energy": {
            "moire_potential": {"enabled": False},
            "soc": {"enabled": False},
        }
    }
    workflow = MoireEvalWorkflow.from_config(cfg)
    workflow.build_runtime()

    rng = jax.random.key(0)
    batched_data = data_init(workflow.system, 1, rng)
    params = workflow.wf.init_params(batched_data.unbatched_example(), rng)
    est_pipeline = workflow.evaluation_stage.estimators
    state = est_pipeline.init(batched_data, rng)
    step_stats, _ = est_pipeline.evaluate(params, batched_data, state, rng)

    assert "energy:moire_potential" not in step_stats
    assert "energy:soc" not in step_stats
    assert step_stats["total_energy"] == pytest.approx(
        step_stats["energy:kinetic"] + step_stats["energy:coulomb"]
    )


@pytest.mark.parametrize("workflow_cls", [MoireTrainWorkflow, MoireEvalWorkflow])
def test_singlephase_requires_spin_channels_divisible_by_kmesh(
    tmp_path: Path, workflow_cls
):
    cfg = _base_config(tmp_path)
    cfg["system"]["electron_spins"] = [5, 4]
    cfg["system"]["nelec"] = 9
    cfg["system"]["supercell_matrix"] = [[3, 0], [0, 3]]

    with pytest.raises(ValidationError, match="divisible by nk=9"):
        workflow_cls.from_config(cfg)


@pytest.mark.parametrize("workflow_cls", [MoireTrainWorkflow, MoireEvalWorkflow])
def test_singlephase_rejects_k_com_during_parsing(tmp_path: Path, workflow_cls):
    cfg = _base_config(tmp_path)
    cfg["wf"]["k_com"] = [0.0, 0.0]

    with pytest.raises(
        ValidationError, match="k_com must be None when phase_mode='singlephase'"
    ):
        workflow_cls.from_config(cfg)


@pytest.mark.parametrize("workflow_cls", [MoireTrainWorkflow, MoireEvalWorkflow])
def test_bilayer_estimators_reject_non_bilayer_wavefunction_during_parsing(
    tmp_path: Path, workflow_cls
):
    cfg = _base_config(tmp_path)
    cfg["wf"]["nlayers"] = 3

    with pytest.raises(ValidationError, match="exactly two layers"):
        workflow_cls.from_config(cfg)

    cfg["estimators"] = {"energy": {"moire_potential": {"enabled": False}}}
    with pytest.raises(ValidationError, match="MoireSOC supports exactly two layers"):
        workflow_cls.from_config(cfg)

    cfg["estimators"]["energy"]["soc"] = {"enabled": False}
    workflow_cls.from_config(cfg)


def test_joint_proposal_moves_positions_and_spin_coords_together():
    lattice = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    sampler = MoireJointMCMCSampler(spin_mass=2.0)
    sampler.configure(lattice)
    proposal = sampler.sampling_proposal
    data = {"positions": jnp.zeros((32, 2, 2)), "spin_coords": jnp.zeros((32, 2))}
    out = proposal(jax.random.key(0), data, jnp.array(0.2))
    assert out["positions"].shape == data["positions"].shape
    assert out["spin_coords"].shape == data["spin_coords"].shape
    pos_step = np.asarray(out["positions"] - data["positions"])
    spin_step = np.asarray((out["spin_coords"] + np.pi) % (2.0 * np.pi) - np.pi)
    frac_positions = np.asarray(out["positions"] @ jnp.linalg.inv(lattice))
    assert np.std(pos_step) > 0.0
    assert np.std(spin_step) > 0.0
    assert np.all(frac_positions >= 0.0)
    assert np.all(frac_positions < 1.0)
    assert np.all(np.asarray(out["spin_coords"]) >= 0.0)
    assert np.all(np.asarray(out["spin_coords"]) < 2.0 * np.pi)
    np.testing.assert_allclose(
        np.std(spin_step),
        2.0 * np.pi * 0.2 / 2.0,
        rtol=0.25,
    )
