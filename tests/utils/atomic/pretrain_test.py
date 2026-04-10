# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for pretraining functionality.

Pretraining matches wavefunction orbitals against SCF (Hartree-Fock) orbitals.
These tests verify that the orbital shapes and pretrain loss computation
work correctly for different wavefunction configurations.
"""

import jax
import jax.numpy as jnp
import pytest

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.app.molecule.wavefunction.psiformer import PsiformerWavefunction
from jaqmc.utils.atomic import Atom, make_pretrain_loss
from jaqmc.utils.atomic.scf import MolecularSCF

TEST_KEY = jax.random.PRNGKey(42)


def make_test_data(nspins: tuple[int, int], key: jax.Array = TEST_KEY) -> MoleculeData:
    """Create MoleculeData for testing.

    Returns:
        MoleculeData with electrons, single atom at origin, and spin configuration.
    """
    n_up, n_down = nspins
    n_elec = n_up + n_down
    return MoleculeData(
        electrons=jax.random.normal(key, (n_elec, 3)),
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([float(n_elec)]),
    )


def make_wavefunction(wf_type: str, nspins: tuple[int, int], ndets: int):
    """Create a wavefunction instance for testing.

    Returns:
        A wavefunction instance (FermiNet or Psiformer).

    Raises:
        ValueError: If wf_type is not "ferminet" or "psiformer".
    """
    if wf_type == "ferminet":
        return FermiNetWavefunction(nspins=nspins, ndets=ndets)
    elif wf_type == "psiformer":
        return PsiformerWavefunction(nspins=nspins, ndets=ndets)
    raise ValueError(f"Unknown wavefunction type: {wf_type}")


class TestPretrainOrbitalShape:
    """Tests for orbital shape compatibility with pretraining."""

    @pytest.mark.parametrize("wf_type", ["ferminet", "psiformer"])
    @pytest.mark.parametrize("nspins,ndets", [((2, 1), 4), ((1, 1), 8), ((3, 2), 4)])
    def test_orbital_shape(self, wf_type, nspins, ndets):
        """Test wavefunctions produce orbitals of shape (ndets, N, N)."""
        n_electrons = sum(nspins)
        wf = make_wavefunction(wf_type, nspins, ndets)
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)

        orbitals = wf.orbitals(params, data)
        assert orbitals.shape == (ndets, n_electrons, n_electrons)

    @pytest.mark.parametrize("wf_type", ["ferminet", "psiformer"])
    @pytest.mark.parametrize("nspins", [(2, 1), (1, 1), (3, 2)])
    def test_pretrain_loss_computes(self, wf_type, nspins):
        """Test pretrain loss can be computed."""
        wf = make_wavefunction(wf_type, nspins, ndets=4)
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)

        atoms = [Atom("H", (0.0, 0.0, 0.0))]
        scf = MolecularSCF(atoms, nspins)
        scf.run()

        loss_estimator = make_pretrain_loss(
            orbitals_fn=wf.orbitals, scf=scf, nspins=nspins, full_det=wf.full_det
        )
        loss_estimator.init(data, TEST_KEY)
        stats, _ = loss_estimator.evaluate_local(params, data, {}, None, TEST_KEY)

        assert jnp.isfinite(stats["loss"])
        assert stats["loss"] > 0, "Random params should not match SCF orbitals exactly"
