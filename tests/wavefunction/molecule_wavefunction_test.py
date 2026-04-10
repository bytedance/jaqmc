# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for molecular wavefunction implementations.

Backbone-specific tests are in backbone_test.py.
"""

from dataclasses import replace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.app.molecule import MoleculeTrainWorkflow
from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.app.molecule.wavefunction.psiformer import PsiformerWavefunction
from jaqmc.utils.config import ConfigManager
from jaqmc.wavefunction.output.envelope import EnvelopeType

# Shared test key
TEST_KEY = jax.random.PRNGKey(42)


def make_test_data(nspins: tuple[int, int], key: jax.Array = TEST_KEY) -> MoleculeData:
    """Create MoleculeData for testing with a single atom at origin.

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


class TestAntisymmetry:
    """Tests for wavefunction antisymmetry under electron permutation."""

    @pytest.mark.parametrize(
        "wf_cls",
        [FermiNetWavefunction, PsiformerWavefunction],
        ids=["ferminet", "psiformer"],
    )
    def test_same_spin_swap_flips_sign(self, wf_cls):
        """Swapping two same-spin electrons must flip the sign and preserve |psi|."""
        wf = wf_cls(nspins=(2, 0))

        key = jax.random.PRNGKey(42)
        electrons = jax.random.normal(key, (2, 3))
        data = MoleculeData(electrons, jnp.array([[0.0, 0.0, 0.0]]), jnp.array([2.0]))
        params = wf.init_params(data, key)

        out = wf.apply(params, data)

        data_perm = replace(data, electrons=electrons[::-1])
        out_perm = wf.apply(params, data_perm)

        assert out["sign_logpsi"] == -out_perm["sign_logpsi"]
        assert out["logpsi"] == pytest.approx(out_perm["logpsi"], rel=5e-4)

    @pytest.mark.parametrize(
        "wf_cls",
        [FermiNetWavefunction, PsiformerWavefunction],
        ids=["ferminet", "psiformer"],
    )
    def test_opposite_spin_swap_changes_value(self, wf_cls):
        """Opposite-spin swap must produce a different wavefunction value.

        Antisymmetry is enforced within each spin channel, not across channels.
        Swapping one up-spin electron with one down-spin electron produces a
        different wavefunction value because the orbitals are spin-dependent.

        Unlike same-spin permutations (which guarantee sign flip and magnitude
        preservation), cross-spin swaps have no simple symmetry relation — both
        sign and magnitude may change.
        """
        wf = wf_cls(nspins=(1, 1))

        key = jax.random.PRNGKey(42)
        electrons = jax.random.normal(key, (2, 3))
        data = MoleculeData(electrons, jnp.array([[0.0, 0.0, 0.0]]), jnp.array([2.0]))
        params = wf.init_params(data, key)

        out = wf.apply(params, data)

        data_perm = replace(data, electrons=electrons[::-1])
        out_perm = wf.apply(params, data_perm)

        psi = out["sign_logpsi"] * jnp.exp(out["logpsi"])
        psi_perm = out_perm["sign_logpsi"] * jnp.exp(out_perm["logpsi"])
        assert not jnp.allclose(psi, psi_perm, rtol=1e-3)


class TestInvalidJastrowType:
    """Tests for invalid jastrow type error handling."""

    def test_psiformer_invalid_jastrow_type(self):
        """Test that PsiformerWavefunction raises error for invalid jastrow type."""
        wf = PsiformerWavefunction(nspins=(2, 1), ndets=4, jastrow="invalid_jastrow")
        data = make_test_data((2, 1))

        with pytest.raises(ValueError, match="Invalid jastrow"):
            wf.init_params(data, TEST_KEY)


class TestEdgeCases:
    """Tests for edge cases with nspins configurations."""

    @pytest.mark.parametrize(
        "wf_cls,nspins,wf_kwargs",
        [
            (PsiformerWavefunction, (1, 0), {"ndets": 4, "jastrow": "simple_ee"}),
            (FermiNetWavefunction, (1, 0), {"ndets": 4}),
            (PsiformerWavefunction, (0, 2), {"ndets": 4, "jastrow": "simple_ee"}),
        ],
        ids=["psiformer_1_0", "ferminet_1_0", "psiformer_0_2"],
    )
    def test_edge_nspins(self, wf_cls, nspins, wf_kwargs):
        """Test wavefunction with edge case nspins configurations."""
        wf = wf_cls(nspins=nspins, **wf_kwargs)
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)
        out = wf.apply(params, data)
        assert jnp.isfinite(out["logpsi"])


class TestGradientFlow:
    """Tests for gradient flow through wavefunction components."""

    @pytest.mark.parametrize(
        "wf_cls,nspins,wf_kwargs",
        [
            (
                PsiformerWavefunction,
                (2, 1),
                {"ndets": 4, "jastrow": "simple_ee", "orbitals_spin_split": False},
            ),
            (
                PsiformerWavefunction,
                (2, 1),
                {"ndets": 4, "jastrow": "simple_ee", "orbitals_spin_split": True},
            ),
            (FermiNetWavefunction, (2, 1), {"ndets": 4, "orbitals_spin_split": False}),
            (FermiNetWavefunction, (2, 1), {"ndets": 4, "orbitals_spin_split": True}),
        ],
        ids=[
            "psiformer_split_false",
            "psiformer_split_true",
            "ferminet_split_false",
            "ferminet_split_true",
        ],
    )
    def test_gradient_wrt_electrons(self, wf_cls, nspins, wf_kwargs):
        """Test that output and gradients are finite w.r.t. electron positions."""
        wf = wf_cls(nspins=nspins, **wf_kwargs)
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)

        out = wf.apply(params, data)
        assert jnp.isfinite(out["logpsi"])

        def logpsi_fn(electrons):
            return wf.apply(params, replace(data, electrons=electrons))["logpsi"]

        grad_electrons = jax.grad(logpsi_fn)(data.electrons)
        assert jnp.all(jnp.isfinite(grad_electrons))

    def test_psiformer_hessian(self):
        """Test Psiformer Hessian (second derivatives) is finite."""
        nspins = (2, 1)
        wf = PsiformerWavefunction(nspins=nspins, ndets=4, jastrow="simple_ee")
        data = make_test_data(nspins)
        params = wf.init_params(data, TEST_KEY)

        def logpsi_fn_flat(electrons_flat):
            new_data = replace(data, electrons=electrons_flat.reshape(-1, 3))
            return wf.apply(params, new_data)["logpsi"]

        hess = jax.hessian(logpsi_fn_flat)(data.electrons.ravel())
        assert jnp.all(jnp.isfinite(hess))


def make_workflow_config(tmp_path, wf_config: dict) -> ConfigManager:
    """Create a minimal workflow config for testing.

    Returns:
        ConfigManager with H2 molecule system and minimal iteration counts.
    """
    return ConfigManager(
        {
            "workflow": {"seed": 42, "save_path": str(tmp_path), "batch_size": 64},
            "wf": wf_config,
            "system": {
                "electron_spins": [1, 1],
                "atoms": [
                    {"symbol": "H", "coords": [1, 0, 0]},
                    {"symbol": "H", "coords": [-1, 0, 0]},
                ],
            },
            "pretrain": {"run": {"iterations": 2, "burn_in": 0}},
            "train": {
                "run": {"iterations": 10, "burn_in": 10},
                "optim": {"learning_rate": {"rate": 0.01}},
            },
        }
    )


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for full training workflow with different wavefunctions."""

    @pytest.mark.parametrize(
        "wf_config",
        [
            {
                "module": "jaqmc.app.molecule.wavefunction.psiformer",
                "num_layers": 1,
                "num_heads": 2,
                "heads_dim": 8,
            },
            {
                "module": "jaqmc.app.molecule.wavefunction.ferminet",
                "hidden_dims_single": [16, 16],
                "hidden_dims_double": [4, 4],
            },
        ],
        ids=["psiformer", "ferminet"],
    )
    def test_workflow_runs(self, tmp_path, wf_config):
        """Test molecule workflow completes without error."""
        cfg = make_workflow_config(tmp_path, wf_config)
        MoleculeTrainWorkflow(cfg)()

        # Verify training stats file was created with finite, improving losses
        stats_file = tmp_path / "train_stats.h5"
        assert stats_file.exists(), "Training stats file was not created"
        with h5py.File(stats_file, "r") as f:
            losses = f["loss"][:]
        assert np.all(np.isfinite(losses)), "Training produced non-finite loss values"
        assert np.mean(losses[-3:]) < np.mean(losses[:3]), (
            f"Training loss did not decrease: first 3 mean={np.mean(losses[:3]):.4f}, "
            f"last 3 mean={np.mean(losses[-3:]):.4f}"
        )


# Flaky: Stochasity due to spherical averaging uses MC sampling with random directions.
@pytest.mark.flaky
class TestCuspCondition:
    """Tests for electron-electron cusp conditions via spherical averaging.

    The cusp condition requires:
        lim_{r→0} ⟨∂(log ψ)/∂r⟩_Ω = c

    where c = 0.5 for antiparallel spins, and ⟨...⟩_Ω denotes spherical average.
    """

    @staticmethod
    def sample_unit_sphere(key: jax.Array, n_samples: int) -> jax.Array:
        """Sample random unit vectors uniformly on the sphere.

        Returns:
            Array of shape (n_samples, 3) with unit vectors.
        """
        raw = jax.random.normal(key, (n_samples, 3))
        return raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)

    @staticmethod
    def radial_derivative_logpsi(
        wf: PsiformerWavefunction, params, data: MoleculeData, i: int, j: int
    ) -> jax.Array:
        r"""Compute ∂(log ψ)/∂r_{ij} where r_{ij} = |r_i - r_j|.

        Returns:
            Scalar radial derivative of log wavefunction.
        """
        electrons = data.electrons
        r_i, r_j = electrons[i], electrons[j]
        r_ij = jnp.linalg.norm(r_i - r_j)
        n_hat = (r_i - r_j) / r_ij

        def logpsi_fn(elec):
            return wf.apply(params, replace(data, electrons=elec))["logpsi"]

        grad = jax.grad(logpsi_fn)(electrons)
        return jnp.dot(grad[i], n_hat)

    @staticmethod
    def make_two_electron_data(
        distance: float,
        direction: jax.Array,
        center: jax.Array,
    ) -> MoleculeData:
        """Create MoleculeData with two electrons at given distance.

        Returns:
            MoleculeData with one up-spin and one down-spin electron.
        """
        r_0 = center + (distance / 2) * direction  # up spin
        r_1 = center - (distance / 2) * direction  # down spin

        return MoleculeData(
            electrons=jnp.stack([r_0, r_1]),
            atoms=jnp.array([[0.0, 0.0, 0.0]]),
            charges=jnp.array([2.0]),
        )

    def spherical_avg_radial_derivative(
        self,
        wf: PsiformerWavefunction,
        params,
        data_maker,
        electron_indices: tuple[int, int],
        n_samples: int,
        key: jax.Array,
    ) -> jax.Array:
        """Compute spherically-averaged radial derivative.

        Args:
            wf: Wavefunction to evaluate.
            params: Wavefunction parameters.
            data_maker: Callable that takes a direction vector and returns MoleculeData.
            electron_indices: Pair (i, j) of electron indices for the radial derivative.
            n_samples: Number of random directions to sample.
            key: JAX random key.

        Returns:
            Mean radial derivative averaged over random directions.
        """
        directions = self.sample_unit_sphere(key, n_samples)
        i, j = electron_indices

        @jax.vmap
        def get_deriv(direction):
            data = data_maker(direction)
            return self.radial_derivative_logpsi(wf, params, data, i, j)

        derivs = get_deriv(directions)
        return jnp.mean(derivs)

    def _make_two_electron_data_maker(self, test_distance, center):
        """Create data maker for 2-electron configuration.

        Returns:
            Callable that takes a direction and returns MoleculeData.
        """

        def data_maker(direction):
            return self.make_two_electron_data(test_distance, direction, center)

        return data_maker

    def _make_three_electron_data_maker(
        self, test_distance, center, third_electron_pos
    ):
        """Create data maker for 3-electron configuration.

        Returns:
            Callable that takes a direction and returns MoleculeData.
        """

        def data_maker(direction):
            r_0 = center + (test_distance / 2) * direction  # first up spin
            r_1 = third_electron_pos  # second up spin (away from cusp pair)
            r_2 = center - (test_distance / 2) * direction  # down spin
            return MoleculeData(
                electrons=jnp.stack([r_0, r_1, r_2]),
                atoms=jnp.array([[0.0, 0.0, 0.0]]),
                charges=jnp.array([3.0]),
            )

        return data_maker

    @pytest.mark.parametrize(
        "nspins,electron_indices,n_electrons",
        [
            ((1, 1), (0, 1), 2),
            ((2, 1), (0, 2), 3),
        ],
        ids=["2e_antiparallel", "3e_antiparallel"],
    )
    @pytest.mark.parametrize("orbitals_spin_split", [False, True])
    @pytest.mark.parametrize("jastrow_alpha_init", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize(
        "envelope",
        [EnvelopeType.isotropic, EnvelopeType.abs_isotropic, EnvelopeType.diagonal],
    )
    def test_antiparallel_cusp(
        self,
        nspins,
        electron_indices,
        n_electrons,
        orbitals_spin_split,
        jastrow_alpha_init,
        envelope,
    ):
        """Test antiparallel cusp condition holds for various parameter settings.

        The spherically-averaged radial derivative at small r should equal
        the cusp value c=0.5 for antiparallel spins, regardless of:
        - Number of electrons (2 or 3)
        - orbitals_spin_split setting
        - jastrow_alpha_init value
        - envelope type
        """
        # Use smaller network to reduce orbital gradient noise
        wf = PsiformerWavefunction(
            nspins=nspins,
            ndets=4,
            num_layers=1,
            heads_dim=16,
            mlp_hidden_dims=[64],
            jastrow="simple_ee",
            orbitals_spin_split=orbitals_spin_split,
            jastrow_alpha_init=jastrow_alpha_init,
            envelope=envelope,
        )

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        init_data = make_test_data(nspins, key1)
        params = wf.init_params(init_data, key2)

        n_samples = 1000
        center = jnp.array([2.0, 0.0, 0.0])  # away from nucleus
        test_distance = 0.01

        if n_electrons == 2:
            data_maker = self._make_two_electron_data_maker(test_distance, center)
        else:
            third_electron_pos = jnp.array([-3.0, 1.0, 0.5])
            data_maker = self._make_three_electron_data_maker(
                test_distance, center, third_electron_pos
            )

        avg_deriv = self.spherical_avg_radial_derivative(
            wf, params, data_maker, electron_indices, n_samples, key3
        )

        expected_cusp = 0.5
        assert jnp.isclose(avg_deriv, expected_cusp, atol=0.1), (
            f"Expected cusp ~{expected_cusp}, got {avg_deriv}"
        )
