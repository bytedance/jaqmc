# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from jax import numpy as jnp

from jaqmc.app.molecule.config.base import MoleculeConfig
from jaqmc.app.molecule.workflow import make_estimators as make_molecule_estimators
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.ph import PHEnergy
from jaqmc.utils.atomic import Atom
from jaqmc.utils.config import ConfigManager


class _DummyWavefunction:
    full_det = True

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
    wf = _DummyWavefunction()
    scf = SimpleNamespace(_mol=SimpleNamespace(_ecp={"Li": object()}))
    system_config = MoleculeConfig(
        atoms=[
            Atom("Fe", [0.0, 0.0, 0.0], charge=16),
            Atom("Li", [2.0, 0.0, 0.0], charge=1),
            Atom("H", [0.0, 2.0, 0.0]),
        ],
        electron_spins=(9, 9),
        pp={"Li": "ccecp", "Fe": "ph"},
    )

    estimators = make_molecule_estimators(cfg, wf, scf, system_config, True)

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
    wf = _DummyWavefunction()
    scf = SimpleNamespace(_mol=SimpleNamespace(_ecp={}))
    system_config = MoleculeConfig(
        atoms=[Atom("Fe", [0.0, 0.0, 0.0], charge=16)],
        electron_spins=(8, 8),
        pp={"Fe": "ph"},
    )

    make_molecule_estimators(cfg, wf, scf, system_config, True)

    with pytest.raises(SystemExit):
        cfg.finalize(raise_on_unused=True)
