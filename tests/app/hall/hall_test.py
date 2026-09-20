# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the quantum Hall workflow components."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall import HallEvalWorkflow, HallTrainWorkflow
from jaqmc.app.hall.config import HallConfig, InteractionType
from jaqmc.app.hall.data import HallData, data_init
from jaqmc.app.hall.estimator.penalized_loss import PenalizedLoss
from jaqmc.app.hall.hamiltonian import SpherePotential
from jaqmc.app.hall.wavefunction.free import Free
from jaqmc.app.hall.wavefunction.jastrow import SphericalJastrow
from jaqmc.app.hall.wavefunction.laughlin import Laughlin
from jaqmc.app.hall.wavefunction.mhpo import MHPO
from jaqmc.estimator.angular_momentum import SphericalAngularMomentum
from jaqmc.estimator.kinetic import LaplacianMode, SphericalKinetic
from jaqmc.geometry.sphere import (
    cartesian_from_spinor,
    sphere_proposal,
    spinor_coordinates_from_angles,
)
from jaqmc.laplacian import forward_laplacian, make_laplacian_input
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.wiring import wire


def _sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def _cartesian(electrons):
    u, v = spinor_coordinates_from_angles(electrons)
    return cartesian_from_spinor(u, v)


def _make_lll_from_spinor(nelec: int, Q: int):
    def log_psi(_, u, v):
        lll_orb = jnp.stack(
            [u**m * v ** (2 * Q - m) for m in range(nelec)],
            axis=-1,
        )
        sign, logdet = jnp.linalg.slogdet(lll_orb)
        return logdet + jnp.log(sign)

    return log_psi


def _eval_single(estimator, data):
    return estimator.evaluate_single_walker({}, data, {}, None, jax.random.PRNGKey(0))[
        0
    ]


def _kinetic_near_pole(estimator, params, electrons, theta):
    """Local kinetic energies with electron 0 at ``theta`` over probe phis."""
    local_energies = []
    for phi in (0.0, 1.3, -2.1):
        near_pole = electrons.at[0].set(jnp.array([theta, phi]))
        stats, _ = estimator.evaluate_single_walker(
            params,
            HallData(electrons=near_pole),
            {},
            None,
            jax.random.PRNGKey(0),
        )
        local_energies.append(stats["energy:kinetic"])
    return np.asarray(local_energies)


class TestHallData:
    def test_data_init_shapes(self):
        cfg = HallConfig(flux=2, nspins=(3, 0))
        batched = data_init(cfg, size=16, rngs=jax.random.PRNGKey(0))
        assert batched.data.electrons.shape == (16, 3, 2)

    def test_data_init_ranges(self):
        cfg = HallConfig(flux=4, nspins=(2, 1))
        batched = data_init(cfg, size=32, rngs=jax.random.PRNGKey(1))
        theta = batched.data.electrons[..., 0]
        phi = batched.data.electrons[..., 1]
        assert jnp.all(theta >= 0) and jnp.all(theta <= jnp.pi)
        assert jnp.all(phi >= -jnp.pi) and jnp.all(phi <= jnp.pi)


class TestSphereProposal:
    def test_stays_on_sphere(self):
        key = jax.random.PRNGKey(42)
        x = _sample(key, 4, 3)
        key, subkey = jax.random.split(key)
        x_new = sphere_proposal(subkey, x, 0.1)
        theta = x_new[..., 0]
        assert jnp.all(theta >= 0) and jnp.all(theta <= jnp.pi)

    def test_shape_preserved(self):
        key = jax.random.PRNGKey(42)
        x = _sample(key, 8, 5)
        key, subkey = jax.random.split(key)
        x_new = sphere_proposal(subkey, x, 0.05)
        assert x_new.shape == x.shape


def _requires_forward_laplacian():
    return pytest.mark.skipif(
        jax.__version_info__ < (0, 7, 1),
        reason="forward_laplacian mode requires JAX >= 0.7.1",
    )


SPHERICAL_LAPLACIAN_MODES = (
    LaplacianMode.scan,
    LaplacianMode.fori_loop,
    pytest.param(
        LaplacianMode.forward_laplacian,
        marks=_requires_forward_laplacian(),
        id="forward_laplacian",
    ),
)


class TestSphericalKinetic:
    @pytest.mark.parametrize("mode", SPHERICAL_LAPLACIAN_MODES)
    def test_free_electron(self, mode: LaplacianMode):
        """Spherical harmonics Y_1m: 3 electrons, Q=0, expect KE=3."""

        def log_psi_from_spinor(_, u, v):
            # Filled L=1 shell: orbitals are the Cartesian coordinates.
            cartesian = cartesian_from_spinor(u, v)
            sign, logdet = jnp.linalg.slogdet(cartesian)
            return logdet + jnp.log(sign)

        data_arr = _sample(jax.random.PRNGKey(1898), 2, nelec=3)
        estimator = SphericalKinetic(
            monopole_strength=0.0,
            radius=1.0,
            mode=mode,
            f_log_psi_from_spinor=log_psi_from_spinor,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(data_arr)
        assert jnp.allclose(stats["energy:kinetic"], 3, atol=1e-3)

    @pytest.mark.parametrize(
        "nelec,Q",
        [(1, 1), (3, 1)],
    )
    @pytest.mark.parametrize("mode", SPHERICAL_LAPLACIAN_MODES)
    def test_lll_kinetic_energy(self, nelec: int, Q: int, mode: LaplacianMode):
        data_arr = _sample(jax.random.PRNGKey(1898), 2, nelec)
        log_psi_from_spinor = _make_lll_from_spinor(nelec, Q)
        estimator = SphericalKinetic(
            monopole_strength=float(Q),
            radius=float(jnp.sqrt(Q)),
            mode=mode,
            f_log_psi_from_spinor=log_psi_from_spinor,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(data_arr)
        assert jnp.allclose(stats["energy:kinetic"], nelec / 2, atol=1e-3)


class TestHallAngularMomentumRegistration:
    def test_enabled_by_default(self):
        workflow = HallTrainWorkflow(ConfigManager({}))

        assert isinstance(
            workflow.train_stage.estimators.estimators["angular_momentum"],
            SphericalAngularMomentum,
        )

    def test_can_disable_without_penalty(self):
        workflow = HallTrainWorkflow(
            ConfigManager({"estimators": {"enabled": {"angular_momentum": False}}})
        )

        assert "angular_momentum" not in workflow.train_stage.estimators.estimators

    def test_eval_can_enable_without_energy(self):
        workflow = HallEvalWorkflow(
            ConfigManager({"estimators": {"enabled": {"energy": False}}})
        )

        estimators = workflow.evaluation_stage.estimators.estimators
        assert "kinetic" not in estimators
        assert isinstance(estimators["angular_momentum"], SphericalAngularMomentum)

    @pytest.mark.parametrize("penalty_key", ["lz_penalty", "l2_penalty"])
    def test_penalty_rejects_disabled_energy(self, penalty_key):
        with pytest.raises(
            ValueError,
            match=r"Angular-momentum penalties require "
            r"estimators\.enabled\.energy=true",
        ):
            HallEvalWorkflow(
                ConfigManager(
                    {
                        "system": {penalty_key: 1.0},
                        "estimators": {"enabled": {"energy": False}},
                    }
                )
            )

    @pytest.mark.parametrize("penalty_key", ["lz_penalty", "l2_penalty"])
    def test_penalty_rejects_disabled_angular_momentum(self, penalty_key):
        with pytest.raises(
            ValueError,
            match=r"Angular-momentum penalties require "
            r"estimators\.enabled\.angular_momentum=true",
        ):
            HallTrainWorkflow(
                ConfigManager(
                    {
                        "system": {penalty_key: 1.0},
                        "estimators": {"enabled": {"angular_momentum": False}},
                    }
                )
            )


class TestSpherePotential:
    def test_coulomb_two_electrons(self):
        """Two electrons at opposite poles: distance=2, potential=1/(2R)."""
        estimator = SpherePotential(
            interaction_type=InteractionType.coulomb,
            monopole_strength=1.0,
            radius=1.0,
            interaction_strength=1.0,
        )
        electrons = jnp.array([[0.0, 0.0], [jnp.pi, 0.0]])
        data = HallData(electrons=electrons)
        stats, _ = estimator.evaluate_single_walker(
            {}, data, {}, None, jax.random.PRNGKey(0)
        )
        assert jnp.allclose(stats["energy:potential"], 0.5, atol=1e-5)


class TestPenalizedLoss:
    """PenalizedLoss: pure arithmetic on prev_walker_stats."""

    def test_no_penalty(self):
        """With zero penalties, loss == total_energy."""
        est = PenalizedLoss(lz_penalty=0.0, l2_penalty=0.0)
        stats = {"total_energy": 5.0}
        out, _ = est.evaluate_single_walker(
            {},
            HallData(electrons=jnp.zeros((1, 2))),
            stats,
            None,
            jax.random.PRNGKey(0),
        )
        np.testing.assert_allclose(out["penalized_loss"], 5.0)

    def test_lz_penalty_only(self):
        """lz_penalty adds (Lz - center)^2 term."""
        est = PenalizedLoss(lz_center=1.0, lz_penalty=2.0, l2_penalty=0.0)
        stats = {
            "total_energy": 10.0,
            "angular_momentum_z": 3.0,
            "angular_momentum_z_square": 9.0,
        }
        # penalty = 2.0 * (9 - 2*1*3 + 1^2) = 2.0 * 4 = 8
        out, _ = est.evaluate_single_walker(
            {},
            HallData(electrons=jnp.zeros((1, 2))),
            stats,
            None,
            jax.random.PRNGKey(0),
        )
        np.testing.assert_allclose(out["penalized_loss"], 18.0)

    def test_both_penalties(self):
        """Both lz and l2 penalties contribute."""
        est = PenalizedLoss(lz_center=0.0, lz_penalty=1.0, l2_penalty=0.5)
        stats = {
            "total_energy": 1.0,
            "angular_momentum_z": 2.0,
            "angular_momentum_z_square": 4.0,
            "angular_momentum_square": 6.0,
        }
        out, _ = est.evaluate_single_walker(
            {},
            HallData(electrons=jnp.zeros((1, 2))),
            stats,
            None,
            jax.random.PRNGKey(0),
        )
        # energy(1) + lz_penalty(4) + l2_penalty(3) = 8
        np.testing.assert_allclose(out["penalized_loss"], 8.0)


class TestSphericalJastrow:
    """SphericalJastrow: symmetric under same-spin swaps."""

    @pytest.mark.x64_modes
    def test_parameters_are_float32(self, x64_mode):
        input_dtype = jnp.float64 if x64_mode else jnp.float32
        cartesian = _cartesian(
            _sample(jax.random.PRNGKey(0), 1, 3)[0].astype(input_dtype)
        )
        params = SphericalJastrow(nspins=(2, 1)).init(jax.random.PRNGKey(1), cartesian)

        assert all(param.dtype == jnp.float32 for param in jax.tree.leaves(params))

    def test_all_same_spin(self):
        """All electrons same spin: parallel pairs only, no antiparallel."""
        jastrow = SphericalJastrow(nspins=(3, 0))
        cartesian = _cartesian(_sample(jax.random.PRNGKey(0), 1, 3)[0])
        params = jastrow.init(jax.random.PRNGKey(1), cartesian)
        out: jax.Array = jastrow.apply(params, cartesian)  # type: ignore[assignment]
        assert jnp.isfinite(out)

    def test_mixed_spins(self):
        """Mixed spins: both parallel and antiparallel pairs."""
        jastrow = SphericalJastrow(nspins=(2, 1))
        cartesian = _cartesian(_sample(jax.random.PRNGKey(0), 1, 3)[0])
        params = jastrow.init(jax.random.PRNGKey(1), cartesian)
        out: jax.Array = jastrow.apply(params, cartesian)  # type: ignore[assignment]
        assert jnp.isfinite(out)

    def test_one_per_spin(self):
        """One electron per spin: no parallel pairs, only antiparallel."""
        jastrow = SphericalJastrow(nspins=(1, 1))
        cartesian = _cartesian(_sample(jax.random.PRNGKey(0), 1, 2)[0])
        params = jastrow.init(jax.random.PRNGKey(1), cartesian)
        out: jax.Array = jastrow.apply(params, cartesian)  # type: ignore[assignment]
        assert jnp.isfinite(out)

    def test_symmetric_under_same_spin_swap(self):
        """Jastrow is symmetric: swapping two same-spin electrons is invariant."""
        jastrow = SphericalJastrow(nspins=(3, 0))
        electrons = _sample(jax.random.PRNGKey(7), 1, 3)[0]
        cartesian = _cartesian(electrons)
        params = jastrow.init(jax.random.PRNGKey(1), cartesian)
        original: jax.Array = jastrow.apply(params, cartesian)  # type: ignore[assignment]
        e_swap = electrons.at[0].set(electrons[1]).at[1].set(electrons[0])
        swapped: jax.Array = jastrow.apply(params, _cartesian(e_swap))  # type: ignore[assignment]
        np.testing.assert_allclose(float(original), float(swapped), atol=1e-5)


class TestFreeWavefunction:
    """Free wavefunction: antisymmetry and exact kinetic energy."""

    def _make_free(self, nspins, flux):
        wf = Free()
        wire(wf, nspins=nspins, flux=flux)
        return wf

    def test_antisymmetry(self):
        """Swapping two same-spin electrons flips the sign of psi."""
        wf = self._make_free(nspins=(3, 0), flux=2)
        electrons = _sample(jax.random.PRNGKey(0), 1, 3)[0]
        data = HallData(electrons=electrons)
        params = wf.init_params(data, jax.random.PRNGKey(1))

        lp_orig = wf.evaluate(params, data)["logpsi"]

        swapped = electrons.at[0].set(electrons[1]).at[1].set(electrons[0])
        lp_swap = wf.evaluate(params, HallData(electrons=swapped))["logpsi"]

        # psi(swap) / psi(orig) should be -1
        ratio = jnp.exp(lp_swap - lp_orig)
        np.testing.assert_allclose(float(jnp.real(ratio)), -1.0, atol=1e-4)
        np.testing.assert_allclose(float(jnp.imag(ratio)), 0.0, atol=1e-4)

    @_requires_forward_laplacian()
    def test_lll_kinetic_energy(self):
        """Free wf filling LLL: kinetic energy per electron is exactly 1/2."""
        nspins = (3, 0)
        flux = 4
        wf = self._make_free(nspins=nspins, flux=flux)
        electrons = _sample(jax.random.PRNGKey(42), 4, 3)
        data = HallData(electrons=electrons[0])
        wf.init_params(data, jax.random.PRNGKey(1))

        estimator = SphericalKinetic(
            monopole_strength=float(flux / 2),
            radius=float(jnp.sqrt(flux / 2)),
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(electrons)
        np.testing.assert_allclose(stats["energy:kinetic"], sum(nspins) / 2, atol=1e-3)

    def test_overflow_branch(self):
        """Free wf with more electrons than LLL orbitals (fills next LL)."""
        wf = self._make_free(nspins=(2, 0), flux=0)
        electrons = _sample(jax.random.PRNGKey(0), 1, 2)[0]
        data = HallData(electrons=electrons)
        params = wf.init_params(data, jax.random.PRNGKey(1))
        out = wf.evaluate(params, data)
        assert jnp.isfinite(out["logpsi"])


class TestLaughlinWavefunction:
    """Laughlin wavefunction: filling validation and exact kinetic energy."""

    def _make_laughlin(self, nspins, flux, flux_per_elec=1, excitation_lz=0):
        wf = Laughlin(flux_per_elec=flux_per_elec, excitation_lz=excitation_lz)
        wire(wf, nspins=nspins, flux=flux)
        return wf

    def test_unsupported_filling(self):
        """Reject electron counts that do not match the ground-state filling."""
        wf = self._make_laughlin(nspins=(2, 0), flux=6)
        electrons = _sample(jax.random.PRNGKey(0), 1, 2)[0]
        data = HallData(electrons=electrons)
        with pytest.raises(ValueError, match="Unsupported Laughlin filling"):
            wf.init_params(data, jax.random.PRNGKey(1))

    @pytest.mark.parametrize("nelec,Q", [(3, 3), (4, 4.5)])
    @_requires_forward_laplacian()
    def test_kinetic_energy(self, nelec, Q):
        """Laughlin ground state reproduces exact kinetic energy."""
        flux = int(2 * Q)
        wf = self._make_laughlin(nspins=(nelec, 0), flux=flux)
        electrons = _sample(jax.random.PRNGKey(1898), 2, nelec)
        data = HallData(electrons=electrons[0])
        wf.init_params(data, jax.random.PRNGKey(1))

        estimator = SphericalKinetic(
            monopole_strength=float(Q),
            radius=float(jnp.sqrt(Q)),
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(electrons)
        np.testing.assert_allclose(stats["energy:kinetic"], nelec / 2, atol=1e-3)

    @pytest.mark.parametrize("nelec,Q", [(3, 3), (4, 4.5)])
    def test_ground_state_angular_momentum(self, nelec, Q):
        """Laughlin ground state has zero total angular momentum."""
        flux = int(2 * Q)
        wf = self._make_laughlin(nspins=(nelec, 0), flux=flux)
        electrons = _sample(jax.random.PRNGKey(1898), 2, nelec)
        data = HallData(electrons=electrons[0])
        wf.init_params(data, jax.random.PRNGKey(1))
        estimator = SphericalAngularMomentum(
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )
        stats = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )(electrons)

        np.testing.assert_allclose(stats["angular_momentum_z"], 0, atol=1e-3)
        np.testing.assert_allclose(stats["angular_momentum_z_square"], 0, atol=1e-3)
        np.testing.assert_allclose(stats["angular_momentum_square"], 0, atol=1e-3)

    @pytest.mark.parametrize(
        "nelec,flux,excitation_lz",
        [(4, 10, 2), (6, 14, 1)],
    )
    @_requires_forward_laplacian()
    def test_excitation_kinetic_energy(self, nelec, flux, excitation_lz):
        """Quasihole and quasiparticle Laughlin states have exact kinetic energy."""
        Q = flux / 2
        wf = self._make_laughlin(
            nspins=(nelec, 0), flux=flux, excitation_lz=excitation_lz
        )
        electrons = _sample(jax.random.PRNGKey(1898), 2, nelec)
        data = HallData(electrons=electrons[0])
        wf.init_params(data, jax.random.PRNGKey(1))

        estimator = SphericalKinetic(
            monopole_strength=float(Q),
            radius=float(jnp.sqrt(Q)),
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(electrons)
        np.testing.assert_allclose(stats["energy:kinetic"], nelec / 2, atol=1e-3)

    @pytest.mark.parametrize(
        "nelec,flux,excitation_lz",
        [(4, 10, 2), (6, 14, 1)],
    )
    def test_excitation_angular_momentum(self, nelec, flux, excitation_lz):
        """Laughlin excitations reproduce their specified angular momentum."""
        wf = self._make_laughlin(
            nspins=(nelec, 0), flux=flux, excitation_lz=excitation_lz
        )
        electrons = _sample(jax.random.PRNGKey(1898), 2, nelec)
        data = HallData(electrons=electrons[0])
        wf.init_params(data, jax.random.PRNGKey(1))
        estimator = SphericalAngularMomentum(
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )
        stats = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )(electrons)

        np.testing.assert_allclose(
            stats["angular_momentum_z"], excitation_lz, atol=1e-3
        )

    @_requires_forward_laplacian()
    def test_kinetic_stable_near_pole(self):
        """Stereographic kinetic stays on the exact Laughlin value near a pole.

        Float32 measurements for ``nelec=3``, ``Q=3`` keep
        ``max|E - 1.5|`` around ``1e-6`` down to ``theta=1e-7``; the
        tolerances below leave a few times headroom on that path.
        """
        nelec = 3
        Q = 3.0
        exact = nelec / 2
        wf = self._make_laughlin(nspins=(nelec, 0), flux=int(2 * Q))
        electrons = _sample(jax.random.PRNGKey(1898), 1, nelec)[0]
        wf.init_params(HallData(electrons=electrons), jax.random.PRNGKey(1))
        estimator = SphericalKinetic(
            monopole_strength=float(Q),
            radius=float(jnp.sqrt(Q)),
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )

        for theta in (1e-3, 1e-5, 1e-7):
            local_energies = _kinetic_near_pole(estimator, {}, electrons, theta)
            assert np.all(np.isfinite(local_energies))
            np.testing.assert_allclose(local_energies, exact, rtol=0, atol=5e-6)
            np.testing.assert_allclose(
                local_energies,
                local_energies[0],
                rtol=0,
                atol=5e-6,
            )


class TestMHPO:
    """MHPO wavefunction: antisymmetry and composite fermion branch."""

    def _make_mhpo(self, nspins=(2, 1), flux=4, flux_per_elec=0):
        wf = MHPO(ndets=1, num_heads=2, heads_dim=8, num_layers=1)
        wire(wf, nspins=nspins, monopole_strength=flux / 2, flux=flux)
        wf.flux_per_elec = flux_per_elec
        electrons = _sample(jax.random.PRNGKey(0), 1, sum(nspins))[0]
        data = HallData(electrons=electrons)
        params = wf.init_params(data, jax.random.PRNGKey(1))
        return wf, params, data, electrons

    def test_antisymmetry(self):
        """Swapping two same-spin electrons flips the sign."""
        wf, params, data, electrons = self._make_mhpo(nspins=(3, 0), flux=4)
        lp_orig = wf.evaluate(params, data)["logpsi"]

        swapped = electrons.at[0].set(electrons[1]).at[1].set(electrons[0])
        lp_swap = wf.evaluate(params, HallData(electrons=swapped))["logpsi"]

        ratio = jnp.exp(lp_swap - lp_orig)
        np.testing.assert_allclose(float(jnp.real(ratio)), -1.0, atol=1e-3)
        np.testing.assert_allclose(float(jnp.imag(ratio)), 0.0, atol=1e-3)

    def test_phi_periodicity(self):
        """logpsi(phi) == logpsi(phi + 2*pi) for any electron."""
        wf, params, data, electrons = self._make_mhpo()
        lp_orig = wf.evaluate(params, data)["logpsi"]

        shifted = electrons.at[0, 1].add(2 * jnp.pi)
        lp_shift = wf.evaluate(params, HallData(electrons=shifted))["logpsi"]
        np.testing.assert_allclose(
            float(jnp.real(lp_orig)),
            float(jnp.real(lp_shift)),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(jnp.imag(lp_orig)),
            float(jnp.imag(lp_shift)),
            atol=1e-4,
        )

    def test_pole_phi_independence(self):
        r"""At a pole, \|psi\|^2 must not depend on phi.

        MHPO has a neural backbone, so we use a looser tolerance than
        Free. The phi-dependent subleading terms scale as theta^2.
        """
        pole_theta = 5e-4
        wf, params, _, electrons = self._make_mhpo()
        phis = jnp.array([0.0, 1.0, 2.5, -1.3])
        re_log_psis = []
        for phi_val in phis:
            e = electrons.at[0].set(jnp.array([pole_theta, phi_val]))
            lp = wf.evaluate(params, HallData(electrons=e))["logpsi"]
            re_log_psis.append(float(jnp.real(lp)))
        np.testing.assert_allclose(re_log_psis, re_log_psis[0], atol=1e-2)

    def test_composite_fermion(self):
        """Composite fermion branch (flux_per_elec > 0) produces finite output."""
        wf, params, data, _ = self._make_mhpo(flux_per_elec=2)
        out = wf.evaluate(params, data)
        assert jnp.isfinite(out["logpsi"])

    @_requires_forward_laplacian()
    def test_sparse_forward_laplacian_mhpo_avoids_gather_and_reduce_max_handlers(
        self, monkeypatch
    ):
        wf, params, data, _ = self._make_mhpo()

        def logpsi_fn(electrons):
            return wf.logpsi(params, HallData(electrons=electrons))

        fl = forward_laplacian(logpsi_fn)
        dense = fl(data.electrons)
        sparse = fl(make_laplacian_input(data.electrons, sparse_axis=0))

        np.testing.assert_allclose(sparse.x, dense.x, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            sparse.dense_jacobian,
            dense.dense_jacobian,
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            sparse.laplacian,
            dense.laplacian,
            rtol=1e-4,
            atol=1e-4,
        )


class TestNeuralHallKineticSmoke:
    @_requires_forward_laplacian()
    def test_mhpo_kinetic_regular_near_pole(self):
        """MHPO local kinetic stays finite and phi-stable near a pole.

        Exactness of the stereographic estimator is covered by Laughlin;
        this only checks that the neural ansatz remains regular enough
        for the kinetic to settle near the pole.
        """
        nspins = (2, 0)
        flux = 3
        wf = MHPO(ndets=1, num_heads=2, heads_dim=8, num_layers=1)
        wire(wf, nspins=nspins, monopole_strength=flux / 2, flux=flux)
        electrons = _sample(jax.random.PRNGKey(81), 1, sum(nspins))[0]
        params = wf.init_params(HallData(electrons=electrons), jax.random.PRNGKey(1))
        estimator = SphericalKinetic(
            monopole_strength=float(flux / 2),
            radius=float(jnp.sqrt(flux / 2)),
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )

        control = _kinetic_near_pole(estimator, params, electrons, 1e-3)
        assert np.all(np.isfinite(control))
        assert np.max(np.abs(control)) < 10

        for theta in (1e-5, 1e-7):
            local_energies = _kinetic_near_pole(estimator, params, electrons, theta)
            assert np.all(np.isfinite(local_energies))
            np.testing.assert_allclose(local_energies, control, rtol=0, atol=5e-2)
            np.testing.assert_allclose(
                local_energies,
                local_energies[0],
                rtol=0,
                atol=1e-3,
            )

    @_requires_forward_laplacian()
    def test_mhpo_kinetic_smoke(self):
        nspins = (2, 1)
        flux = 4
        electrons = _sample(jax.random.PRNGKey(7), 1, sum(nspins))[0]
        data = HallData(electrons=electrons)
        wf = MHPO(ndets=1, num_heads=2, heads_dim=8, num_layers=1)
        wire(wf, nspins=nspins, monopole_strength=flux / 2, flux=flux)
        params = wf.init_params(data, jax.random.PRNGKey(1))
        estimator = SphericalKinetic(
            monopole_strength=float(flux / 2),
            radius=float(jnp.sqrt(flux / 2)),
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )
        stats, _ = estimator.evaluate_single_walker(
            params, data, {}, None, jax.random.PRNGKey(0)
        )
        assert jnp.isfinite(stats["energy:kinetic"])
