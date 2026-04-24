# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-body reduced density matrix (1-RDM) estimator."""

from operator import itemgetter

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.one_rdm import OneRDM, make_monopole_harm
from jaqmc.data import BatchedData


def _sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def _make_lll(nelec: int, Q: float):
    def log_psi(_, data):
        electrons = data["electrons"]
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = jnp.cos(theta / 2) * jnp.exp(1j * phi / 2)
        v = jnp.sin(theta / 2) * jnp.exp(-1j * phi / 2)
        lll_orb = jnp.stack(
            [u**m * v ** (int(2 * Q) - m) for m in range(nelec)], axis=-1
        )
        sign, logdet = jnp.linalg.slogdet(lll_orb)
        return logdet + jnp.log(sign)

    return log_psi


class TestMonopoleHarmonics:
    def test_orthonormality(self):
        """Monopole harmonics should satisfy orthonormality on the sphere."""
        Q = 1.0
        orbitals = [make_monopole_harm(Q, Q, m) for m in np.arange(-Q, Q + 1)]

        n_theta, n_phi = 200, 200
        theta = jnp.linspace(1e-4, jnp.pi - 1e-4, n_theta)
        phi = jnp.linspace(-jnp.pi, jnp.pi, n_phi, endpoint=False)
        theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")
        coords = jnp.stack([theta_grid, phi_grid], axis=-1)

        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        sin_theta = jnp.sin(theta_grid)

        norbs = len(orbitals)
        overlap = np.zeros((norbs, norbs), dtype=complex)
        vals = [orb(coords) for orb in orbitals]
        for i in range(norbs):
            for j in range(norbs):
                integrand = jnp.conj(vals[i]) * vals[j] * sin_theta
                overlap[i, j] = jnp.sum(integrand) * dtheta * dphi

        np.testing.assert_allclose(overlap, np.eye(norbs), atol=0.02)

    def test_Q0_reduces_to_spherical_harmonic(self):
        """For Q=0, l=0, m=0: Y_000 = 1/sqrt(4pi)."""
        Y_000 = make_monopole_harm(0, 0, 0)
        coords = jnp.array([[0.5, 1.0]])
        expected = 1 / jnp.sqrt(4 * jnp.pi)
        assert jnp.allclose(Y_000(coords), expected, atol=1e-6)


class TestOneRDMEvaluateSingleWalker:
    def test_output_shape(self):
        """evaluate_single_walker should return a complex (norbs, norbs) matrix."""
        flux = 2
        nelec = 3
        log_psi = _make_lll(nelec, Q=flux / 2)
        estimator = OneRDM(flux=flux, f_log_psi=log_psi)

        data = HallData(electrons=_sample(jax.random.PRNGKey(0), 1, nelec)[0])
        estimator.init(data, jax.random.PRNGKey(1))

        stats, _ = estimator.evaluate_single_walker(
            None, data, {}, None, jax.random.PRNGKey(2)
        )
        assert "one_rdm" in stats
        norbs = flux + 1
        assert stats["one_rdm"].shape == (norbs, norbs)
        assert jnp.iscomplexobj(stats["one_rdm"])

    def test_jit_compatible(self):
        """evaluate_single_walker should work under jax.jit."""
        flux = 2
        nelec = 3
        log_psi = _make_lll(nelec, Q=flux / 2)
        estimator = OneRDM(flux=flux, f_log_psi=log_psi)

        data = HallData(electrons=_sample(jax.random.PRNGKey(0), 1, nelec)[0])
        estimator.init(data, jax.random.PRNGKey(1))

        jitted = jax.jit(
            lambda d, k: estimator.evaluate_single_walker(None, d, {}, None, k)
        )
        stats, _ = jitted(data, jax.random.PRNGKey(2))
        assert stats["one_rdm"].shape == (flux + 1, flux + 1)


class TestOneRDMSmoke:
    @pytest.fixture
    def setup(self):
        flux = 2
        nelec = 3
        n_walkers = 64
        n_steps = 200
        return {
            "flux": flux,
            "nelec": nelec,
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "log_psi": _make_lll(nelec, Q=flux / 2),
        }

    def test_trace_equals_nelec(self, setup):
        """The trace of the 1-RDM should equal the number of electrons."""
        flux = setup["flux"]
        nelec = setup["nelec"]
        n_walkers = setup["n_walkers"]
        n_steps = setup["n_steps"]
        log_psi = setup["log_psi"]

        estimator = OneRDM(flux=flux, f_log_psi=log_psi)
        sample_data = HallData(electrons=_sample(jax.random.PRNGKey(0), 1, nelec)[0])
        estimator.init(sample_data, jax.random.PRNGKey(1))

        key = jax.random.PRNGKey(42)
        all_step_stats: list[dict] = []

        for step in range(n_steps):
            key, sample_key, eval_key = jax.random.split(key, 3)
            electrons = _sample(sample_key, n_walkers, nelec)
            eval_keys = jax.random.split(eval_key, n_walkers)
            walker_stats, _ = jax.vmap(
                lambda elec, k: estimator.evaluate_single_walker(
                    None, HallData(electrons=elec), {}, None, k
                ),
                in_axes=(0, 0),
            )(electrons, eval_keys)
            # Compute mean manually (reduce needs shard_map context for pmean)
            step_stats = jax.tree.map(lambda x: jnp.nanmean(x, axis=0), walker_stats)
            all_step_stats.append(step_stats)

        batched_stats = jax.tree.map(lambda *xs: jnp.stack(xs), *all_step_stats)
        final = estimator.finalize_stats(batched_stats, None)

        assert "one_rdm" in final
        assert "one_rdm:diagonal" in final
        assert "one_rdm:trace" in final
        assert final["one_rdm"].shape == (flux + 1, flux + 1)
        assert final["one_rdm:diagonal"].shape == (flux + 1,)

        # Trace should be approximately equal to the number of electrons
        trace = final["one_rdm:trace"]
        np.testing.assert_allclose(trace.real, nelec, atol=0.5)
        # Imaginary part of trace should be small
        assert abs(trace.imag) < 0.5


class TestOneRDMPipeline:
    def test_evaluate_batch_walkers_and_finalize_stats(self):
        """OneRDM integrates with the batched evaluator and finalize_stats."""
        flux = 2
        nelec = 3
        n_walkers = 8
        log_psi = _make_lll(nelec, Q=flux / 2)

        estimator = OneRDM(flux=flux, f_log_psi=log_psi)

        electrons = _sample(jax.random.PRNGKey(0), n_walkers, nelec)
        batched_data = BatchedData(
            data=HallData(electrons=electrons),
            fields_with_batch=["electrons"],
        )

        state = estimator.init(batched_data.unbatched_example(), jax.random.PRNGKey(1))

        # evaluate_batch_walkers vmaps evaluate_single_walker over walkers
        walker_stats, state = estimator.evaluate_batch_walkers(
            None, batched_data, {}, state, jax.random.PRNGKey(2)
        )

        norbs = flux + 1
        assert walker_stats["one_rdm"].shape == (n_walkers, norbs, norbs)

        # Manual reduce (skip pmean which needs shard_map context)
        step_stats = jax.tree.map(lambda x: jnp.nanmean(x, axis=0), walker_stats)
        assert step_stats["one_rdm"].shape == (norbs, norbs)

        # Finalize with batch dim of 1
        batched = jax.tree.map(itemgetter(None), step_stats)
        final = estimator.finalize_stats(batched, state)

        assert "one_rdm" in final
        assert "one_rdm:diagonal" in final
        assert "one_rdm:trace" in final
        assert final["one_rdm"].shape == (norbs, norbs)
        assert final["one_rdm:diagonal"].shape == (norbs,)
