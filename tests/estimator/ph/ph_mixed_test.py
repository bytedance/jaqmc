# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import numpy as np
import pyscf.gto
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.ph import PHEnergy
from jaqmc.estimator.total_energy import TotalEnergy


def _gaussian_logpsi(params, data: MoleculeData) -> jnp.ndarray:
    del params
    return -0.3 * jnp.sum(data.electrons**2)


def _gaussian_phase_logpsi(
    params, data: MoleculeData
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.array(1.0), _gaussian_logpsi(params, data)


def _li_ecp_coefficients() -> dict[str, object]:
    mol = pyscf.gto.Mole(atom=[["Li", [0.0, 0.0, 0.0]]], unit="bohr")
    mol.basis = "sto-3g"
    mol.ecp = {"Li": "ccecp"}
    mol.spin = 1
    mol.build()
    return mol._ecp


def test_mixed_ph_ecp_all_electron_path_is_finite():
    """Smoke-test composed PH, ECP, and all-electron local-energy evaluation."""
    data = MoleculeData(
        electrons=jnp.array([[0.4, -0.2, 0.3]]),
        atoms=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        ),
        charges=jnp.array([16.0, 1.0, 1.0]),
    )
    rng = jax.random.PRNGKey(0)

    ph = PHEnergy(
        f_log_psi=_gaussian_logpsi,
        atom_symbols=["Fe", "Li", "H"],
        ph=["Fe"],
    )
    ecp = ECPEnergy(
        ecp_coefficients=_li_ecp_coefficients(),
        atom_symbols=["Fe", "Li", "H"],
        phase_logpsi=_gaussian_phase_logpsi,
    )
    total = TotalEnergy()

    ph.init(data, rng)
    ph_stats, _ = ph.evaluate_single_walker({}, data, {}, None, rng)
    ecp.init(data, rng)
    ecp_stats, _ = ecp.evaluate_single_walker({}, data, ph_stats, None, rng)
    total_stats, _ = total.evaluate_single_walker(
        {},
        data,
        {**ph_stats, **ecp_stats},
        None,
        rng,
    )

    for value in (
        ph_stats["energy:kinetic"],
        ph_stats["energy:ph"],
        ecp_stats["energy:ecp"],
        total_stats["total_energy"],
    ):
        assert jnp.isscalar(value)
        np.testing.assert_allclose(jnp.isfinite(value), True)
