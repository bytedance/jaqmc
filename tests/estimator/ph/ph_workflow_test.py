# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from jax import numpy as jnp

from jaqmc.app.molecule.config import AtomConfig, MoleculeConfig
from jaqmc.app.molecule.workflow import make_estimators as make_molecule_estimators
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.ph import PHEnergy
from jaqmc.utils.config import ConfigManager


class _DummyWavefunction:
    full_det = True

    def init_params(self, data, rngs):
        del data, rngs
        return {}

    def evaluate(self, params, data):
        return {
            "logpsi": self.logpsi(params, data),
            "sign_logpsi": jnp.array(1.0),
        }

    def logpsi(self, params, data):
        del params
        return -jnp.sum(data.electrons**2)

    def phase_logpsi(self, params, data):
        return jnp.array(1.0), self.logpsi(params, data)

    def orbitals(self, params, data):
        del params
        nelec = data.electrons.shape[0]
        return jnp.eye(nelec)[None, ...]


def test_molecule_workflow_wires_ph_parallel_to_ecp():
    """Keep molecule workflows wiring PH as a runtime estimator parallel to ECP."""
    cfg = ConfigManager({})
    wf: Any = _DummyWavefunction()
    system_config = MoleculeConfig(
        atom_configs=[
            AtomConfig(symbol="Fe", coords=[0.0, 0.0, 0.0]),
            AtomConfig(symbol="Li", coords=[2.0, 0.0, 0.0]),
            AtomConfig(symbol="H", coords=[0.0, 2.0, 0.0]),
        ],
        s_z=0,
        pp={"Li": "ccecp", "Fe": "ph"},
    )

    estimators = make_molecule_estimators(
        cfg, wf, system_config, always_enable_energy=True
    )

    assert list(estimators) == ["potential", "ecp", "ph", "total"]
    assert isinstance(estimators["ecp"], ECPEnergy)
    assert isinstance(estimators["ph"], PHEnergy)
    assert estimators["ecp"].atom_symbols == ["Fe", "Li", "H"]
    assert estimators["ph"].atom_symbols == ["Fe", "Li", "H"]
    assert estimators["ph"].ph == ["Fe"]


def test_ph_workflow_rejects_kinetic_mode_override_via_cfg_finalize():
    """PH workflows surface `estimators.energy.kinetic.*` overrides as unused config."""
    cfg = ConfigManager(
        {}, dotlist=["estimators.energy.kinetic.mode=forward_laplacian"]
    )
    wf: Any = _DummyWavefunction()
    system_config = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="Fe", coords=[0.0, 0.0, 0.0])],
        s_z=0,
        pp={"Fe": "ph"},
    )

    make_molecule_estimators(cfg, wf, system_config, always_enable_energy=True)

    with pytest.raises(SystemExit):
        cfg.finalize(raise_on_unused=True)
