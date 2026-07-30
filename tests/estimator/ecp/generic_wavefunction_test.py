# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Generic dense ECP regressions for wavefunctions without cached updates."""

import jax
import numpy as np
import pyscf.gto
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.psiformer import PsiformerWavefunction
from jaqmc.estimator.ecp import ECPEnergy, ECPQuadrature


def _li_ecp_coefficients() -> dict[str, object]:
    molecule = pyscf.gto.Mole(atom=[["Li", [0.0, 0.0, 0.0]]], unit="bohr")
    molecule.basis = "sto-3g"
    molecule.ecp = {"Li": "ccecp"}
    molecule.spin = 1
    molecule.build()
    return molecule._ecp


def test_psiformer_uses_generic_dense_ecp_path():
    """The shared quadrature change does not require a FiRE-specific adapter."""
    key = jax.random.key(7)
    data = MoleculeData(
        electrons=jnp.array([[0.7, -0.2, 0.3]]),
        atoms=jnp.zeros((1, 3)),
        charges=jnp.array([1.0]),
    )
    wavefunction = PsiformerWavefunction(nspins=(1, 0), ndets=1, num_layers=1)
    params = wavefunction.init_params(data, key)

    estimator = ECPEnergy(
        quadrature_id=ECPQuadrature.icosahedron_12,
        phase_logpsi=wavefunction.phase_logpsi,
        ecp_coefficients=_li_ecp_coefficients(),
        atom_symbols=["Li"],
    )
    estimator.init(data, key)
    stats, state = estimator.evaluate_single_walker(params, data, {}, None, key)

    assert state is None
    assert np.isfinite(stats["energy:ecp"])
