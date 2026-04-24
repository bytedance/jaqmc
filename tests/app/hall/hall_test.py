# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the quantum Hall workflow components."""

import importlib.util

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.config import HallConfig, InteractionType
from jaqmc.app.hall.data import HallData, data_init
from jaqmc.app.hall.estimator.penalized_loss import PenalizedLoss
from jaqmc.app.hall.hamiltonian import SpherePotential
from jaqmc.app.hall.wavefunction.free import Free
from jaqmc.app.hall.wavefunction.jastrow import SphericalJastrow
from jaqmc.app.hall.wavefunction.mhpo import MHPO
from jaqmc.estimator.kinetic import LaplacianMode, SphericalKinetic
from jaqmc.geometry.sphere import sphere_proposal
from jaqmc.utils.wiring import wire


def _sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def _make_lll(nelec: int, Q: int):
    def log_psi(_, data):
        electrons = data["electrons"]
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = jnp.cos(theta / 2) * jnp.exp(1j * phi / 2)
        v = jnp.sin(theta / 2) * jnp.exp(-1j * phi / 2)
        lll_orb = jnp.stack([u**m * v ** (2 * Q - m) for m in range(nelec)], axis=-1)
        sign, logdet = jnp.linalg.slogdet(lll_orb)
        return logdet + jnp.log(sign)

    return log_psi


def _eval_single(estimator, data):
    return estimator.evaluate_single_walker(None, data, {}, None, None)[0]


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


def _supports_forward_laplacian() -> bool:
    return (
        jax.__version_info__ >= (0, 7, 1)
        and importlib.util.find_spec("folx") is not None
    )


LAPLACIAN_MODES = [LaplacianMode.scan]
if _supports_forward_laplacian():
    LAPLACIAN_MODES.append(LaplacianMode.forward_laplacian)


class TestSphericalKinetic:
    @pytest.mark.parametrize("mode", LAPLACIAN_MODES)
    def test_free_electron(self, mode):
        """Spherical harmonics Y_1m: 3 electrons, Q=0, expect KE=3."""

        def log_psi(params, data):
            electrons = data["electrons"]
            theta, phi = electrons[..., 0], electrons[..., 1]
            orb = jnp.stack(
                [
                    jnp.sin(theta) * jnp.cos(phi),
                    jnp.cos(theta),
                    jnp.sin(theta) * jnp.sin(phi),
                ],
                axis=-1,
            )
            sign, logdet = jnp.linalg.slogdet(orb)
            return logdet + jnp.log(sign)

        data_arr = _sample(jax.random.PRNGKey(1898), 2, nelec=3)
        estimator = SphericalKinetic(
            mode=mode,
            monopole_strength=0.0,
            radius=1.0,
            f_log_psi=log_psi,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(data_arr)
        assert jnp.allclose(stats["energy:kinetic"], 3, atol=1e-3)
        assert jnp.allclose(stats["angular_momentum_square"], 0, atol=1e-3)

    @pytest.mark.parametrize("mode", LAPLACIAN_MODES)
    @pytest.mark.parametrize(
        "nelec,Q,L_square,L_z",
        [(1, 1, 2, -1), (3, 1, 0, 0)],
    )
    def test_lll_kinetic_and_angular_momentum(
        self, mode, nelec: int, Q: int, L_square: float, L_z: float
    ):
        data_arr = _sample(jax.random.PRNGKey(1898), 2, nelec)
        log_psi = _make_lll(nelec, Q)
        estimator = SphericalKinetic(
            mode=mode,
            monopole_strength=float(Q),
            radius=float(jnp.sqrt(Q)),
            f_log_psi=log_psi,
        )
        batch_eval = jax.jit(
            jax.vmap(
                lambda d: _eval_single(estimator, HallData(electrons=d)),
                in_axes=0,
            )
        )
        stats = batch_eval(data_arr)
        assert jnp.allclose(stats["energy:kinetic"], nelec / 2, atol=1e-3)
        assert jnp.allclose(stats["angular_momentum_z"], L_z, atol=1e-3)
        assert jnp.allclose(stats["angular_momentum_z_square"], L_z**2, atol=1e-3)
        assert jnp.allclose(stats["angular_momentum_square"], L_square, atol=1e-3)


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
        stats, _ = estimator.evaluate_single_walker(None, data, {}, None, None)
        assert jnp.allclose(stats["energy:potential"], 0.5, atol=1e-5)


class TestPenalizedLoss:
    """PenalizedLoss: pure arithmetic on prev_walker_stats."""

    def test_no_penalty(self):
        """With zero penalties, loss == total_energy."""
        est = PenalizedLoss(lz_penalty=0.0, l2_penalty=0.0)
        stats = {"total_energy": 5.0}
        out, _ = est.evaluate_single_walker(None, None, stats, None, None)
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
        out, _ = est.evaluate_single_walker(None, None, stats, None, None)
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
        out, _ = est.evaluate_single_walker(None, None, stats, None, None)
        # energy(1) + lz_penalty(4) + l2_penalty(3) = 8
        np.testing.assert_allclose(out["penalized_loss"], 8.0)


class TestSphericalJastrow:
    """SphericalJastrow: symmetric under same-spin swaps."""

    def test_all_same_spin(self):
        """All electrons same spin: parallel pairs only, no antiparallel."""
        jastrow = SphericalJastrow(nspins=(3, 0))
        electrons = _sample(jax.random.PRNGKey(0), 1, 3)[0]
        params = jastrow.init(jax.random.PRNGKey(1), electrons)
        out = jastrow.apply(params, electrons)
        assert jnp.isfinite(out)

    def test_mixed_spins(self):
        """Mixed spins: both parallel and antiparallel pairs."""
        jastrow = SphericalJastrow(nspins=(2, 1))
        electrons = _sample(jax.random.PRNGKey(0), 1, 3)[0]
        params = jastrow.init(jax.random.PRNGKey(1), electrons)
        out = jastrow.apply(params, electrons)
        assert jnp.isfinite(out)

    def test_one_per_spin(self):
        """One electron per spin: no parallel pairs, only antiparallel."""
        jastrow = SphericalJastrow(nspins=(1, 1))
        electrons = _sample(jax.random.PRNGKey(0), 1, 2)[0]
        params = jastrow.init(jax.random.PRNGKey(1), electrons)
        out = jastrow.apply(params, electrons)
        assert jnp.isfinite(out)

    def test_symmetric_under_same_spin_swap(self):
        """Jastrow is symmetric: swapping two same-spin electrons is invariant."""
        jastrow = SphericalJastrow(nspins=(3, 0))
        electrons = _sample(jax.random.PRNGKey(7), 1, 3)[0]
        params = jastrow.init(jax.random.PRNGKey(1), electrons)
        original = jastrow.apply(params, electrons)
        e_swap = electrons.at[0].set(electrons[1]).at[1].set(electrons[0])
        swapped = jastrow.apply(params, e_swap)
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

    @pytest.mark.parametrize("mode", LAPLACIAN_MODES)
    def test_lll_kinetic_energy(self, mode):
        """Free wf filling LLL: kinetic energy per electron is exactly 1/2."""
        nspins = (3, 0)
        flux = 4
        wf = self._make_free(nspins=nspins, flux=flux)
        electrons = _sample(jax.random.PRNGKey(42), 4, 3)
        data = HallData(electrons=electrons[0])
        wf.init_params(data, jax.random.PRNGKey(1))

        # Free has no trainable params, so wrap logpsi to pass {}
        def log_psi_fn(params, data):
            return wf.logpsi({}, data)

        estimator = SphericalKinetic(
            mode=mode,
            monopole_strength=float(flux / 2),
            radius=float(jnp.sqrt(flux / 2)),
            f_log_psi=log_psi_fn,
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
