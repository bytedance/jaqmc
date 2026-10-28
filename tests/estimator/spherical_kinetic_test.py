# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact-value tests for the spherical kinetic estimator."""

import jax
import pytest
from jax import numpy as jnp

from jaqmc.estimator.kinetic import LaplacianMode, SphericalKinetic
from jaqmc.geometry.sphere import cartesian_from_spinor
from tests.jax_version import requires_forward_laplacian

MODES = (
    LaplacianMode.scan,
    LaplacianMode.fori_loop,
    pytest.param(
        LaplacianMode.forward_laplacian,
        marks=requires_forward_laplacian,
        id="forward_laplacian",
    ),
)


def _sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def _kinetic(estimator, electrons):
    def single(x):
        stats, _ = estimator.evaluate_single_walker(
            {}, {"electrons": x}, {}, None, jax.random.key(0)
        )
        return stats["energy:kinetic"]

    return jax.jit(jax.vmap(single))(electrons)


def _slater_logpsi(orbitals_fn):
    def log_psi(_, u, v):
        sign, logdet = jnp.linalg.slogdet(orbitals_fn(u, v))
        return logdet + jnp.log(sign)

    return log_psi


@pytest.mark.parametrize("mode", MODES)
def test_filled_l1_shell_without_monopole(mode):
    """Filled L=1 shell at Q=0 (orbitals are x, y, z): KE = L(L+1) per electron."""
    estimator = SphericalKinetic(
        monopole_strength=0.0,
        radius=1.0,
        mode=mode,
        f_log_psi_from_spinor=_slater_logpsi(cartesian_from_spinor),
    )
    kinetic = _kinetic(estimator, _sample(jax.random.key(1898), 2, nelec=3))
    assert jnp.allclose(kinetic, 3, atol=1e-3)


@pytest.mark.parametrize("nelec,Q", [(1, 1), (3, 1)])
@pytest.mark.parametrize("mode", MODES)
def test_lowest_landau_level(nelec, Q, mode):
    """LLL Slater determinants have KE = 1/2 per electron with R = sqrt(Q)."""

    def lll_orbitals(u, v):
        return jnp.stack([u**m * v ** (2 * Q - m) for m in range(nelec)], axis=-1)

    estimator = SphericalKinetic(
        monopole_strength=float(Q),
        radius=float(jnp.sqrt(Q)),
        mode=mode,
        f_log_psi_from_spinor=_slater_logpsi(lll_orbitals),
    )
    kinetic = _kinetic(estimator, _sample(jax.random.key(1898), 2, nelec))
    assert jnp.allclose(kinetic, nelec / 2, atol=1e-3)
