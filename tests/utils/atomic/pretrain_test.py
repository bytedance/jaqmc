# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for pretraining functionality.

Pretraining matches wavefunction orbitals against a prepared orbital reference.
These tests verify that the orbital shapes and pretrain loss computation
work correctly for different wavefunction configurations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.app.molecule.wavefunction.lapnet import LapNetWavefunction
from jaqmc.app.molecule.wavefunction.psiformer import PsiformerWavefunction
from jaqmc.estimator import FunctionEstimator
from jaqmc.utils.atomic import make_pretrain_loss

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
        A wavefunction instance.

    Raises:
        ValueError: If wf_type is unknown.
    """
    if wf_type == "ferminet":
        return FermiNetWavefunction(nspins=nspins, ndets=ndets)
    elif wf_type == "psiformer":
        return PsiformerWavefunction(nspins=nspins, ndets=ndets)
    elif wf_type == "lapnet":
        return LapNetWavefunction(
            nspins=nspins,
            ndets=ndets,
            num_layers=1,
            num_heads=2,
            heads_dim=8,
        )
    raise ValueError(f"Unknown wavefunction type: {wf_type}")


class TestPretrainOrbitalShape:
    """Tests for orbital shape compatibility with pretraining."""

    @pytest.mark.parametrize("wf_type", ["ferminet", "psiformer", "lapnet"])
    @pytest.mark.parametrize("nspins,ndets", [((2, 1), 4), ((1, 1), 8), ((3, 2), 4)])
    def test_orbital_shape(self, wf_type, nspins, ndets):
        """Test wavefunctions produce orbitals of shape (ndets, N, N)."""
        n_electrons = sum(nspins)
        wf = make_wavefunction(wf_type, nspins, ndets)
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)

        orbitals = wf.orbitals(params, data)
        assert orbitals.shape == (ndets, n_electrons, n_electrons)

    @pytest.mark.parametrize("wf_type", ["ferminet", "psiformer", "lapnet"])
    @pytest.mark.parametrize("nspins", [(2, 1), (1, 1), (3, 2), (1, 0)])
    def test_pretrain_loss_computes(self, wf_type, nspins):
        """Test pretrain loss can be computed."""
        wf = make_wavefunction(wf_type, nspins, ndets=4)
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)

        class FakeReference:
            def eval_orbitals(self, pos, nspins):
                n_alpha, n_beta = nspins
                leading = pos.shape[:-2]
                return (
                    jnp.ones((*leading, n_alpha, n_alpha)),
                    jnp.ones((*leading, n_beta, n_beta)),
                )

        loss_estimator = make_pretrain_loss(
            orbitals_fn=wf.orbitals,
            orbital_ref=FakeReference(),
            nspins=nspins,
            full_det=wf.full_det,
        )
        loss_estimator.init(data, TEST_KEY)
        assert isinstance(loss_estimator, FunctionEstimator)
        stats, _ = loss_estimator.evaluate_single_walker(
            params, data, {}, None, TEST_KEY
        )

        assert jnp.isfinite(stats["loss"])
        assert stats["loss"] > 0, "Random params should not match reference orbitals"


def test_pretrain_loss_uses_spin_block_diagonal_target():
    class Reference:
        def eval_orbitals(self, pos, nspins):
            del pos, nspins
            return (
                jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
                jnp.asarray([[5.0]]),
            )

    def orbitals(params, data):
        del params, data
        return jnp.asarray([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [30.0, 40.0, 5.0]])

    estimator = make_pretrain_loss(orbitals, Reference(), (2, 1), full_det=True)
    assert isinstance(estimator, FunctionEstimator)
    stats, _ = estimator.evaluate_single_walker(
        {}, make_test_data((2, 1)), {}, None, TEST_KEY
    )

    np.testing.assert_allclose(stats["loss"], (10**2 + 20**2 + 30**2 + 40**2) / 9)


def test_pretrain_loss_handles_an_empty_spin_block():
    class Reference:
        def eval_orbitals(self, pos, nspins):
            del pos, nspins
            return jnp.asarray([[1.0]]), jnp.zeros((0, 0))

    def orbitals(params, data):
        del params, data
        return jnp.asarray([[2.0]])

    estimator = make_pretrain_loss(orbitals, Reference(), (1, 0), full_det=True)
    assert isinstance(estimator, FunctionEstimator)
    stats, _ = estimator.evaluate_single_walker(
        {}, make_test_data((1, 0)), {}, None, TEST_KEY
    )

    np.testing.assert_allclose(stats["loss"], 1.0)
