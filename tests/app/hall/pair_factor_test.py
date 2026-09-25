# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for reduced pair factors used by Hall flux dressings."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.wavefunction import _pair_factor
from jaqmc.app.hall.wavefunction._dressing import (
    PAIR_SPINOR_MATRIX,
    Dressing,
    DressingConfig,
)
from jaqmc.app.hall.wavefunction._pair_factor import log_pair_factor
from jaqmc.app.hall.wavefunction.mhpo import MHPO
from jaqmc.estimator.kinetic import LaplacianMode, SphericalKinetic
from jaqmc.laplacian import make_laplacian_input
from jaqmc.utils.wiring import wire
from tests.laplacian.helpers import check_with_brute_force

pytestmark = pytest.mark.skipif(
    jax.__version_info__ < (0, 7, 1),
    reason="forward_laplacian mode requires JAX >= 0.7.1",
)


@pytest.fixture
def fallbacks(monkeypatch):
    """Record every fallback the pair-factor rule raises.

    Returns:
        List of fallback messages; empty when the custom rule handled every call.
    """
    messages = []

    class RecordingFallback(_pair_factor.AutoLaplacianFallback):
        def __init__(self, message):
            messages.append(message)
            super().__init__(message)

    monkeypatch.setattr(_pair_factor, "AutoLaplacianFallback", RecordingFallback)
    return messages


def _neural_matrix_pair_factor(nelec, *, track_matrix=True):
    ndets, channels = 2, 2
    key, coeff_key = jax.random.split(jax.random.PRNGKey(17))
    coords = jax.random.normal(key, (nelec, 4), dtype=jnp.float64)
    coefficients = 0.02 * jax.random.normal(
        coeff_key,
        (nelec, ndets, channels, nelec, 2, 2, 2, coords.shape[-1]),
        dtype=jnp.float64,
    )
    fixed_features = jnp.sum(jnp.tanh(coords), axis=0)

    def reduced_pair_factor(x):
        z = x[:, :2] + 1j * x[:, 2:]
        features = jnp.sum(jnp.tanh(x), axis=0) if track_matrix else fixed_features
        delta = jnp.einsum("f,idckabrf->idckabr", features, coefficients)
        matrix = delta[..., 0] + 1j * delta[..., 1] + PAIR_SPINOR_MATRIX
        return log_pair_factor(jnp.einsum("ia,idckab->idckb", z, matrix), z)

    return reduced_pair_factor, coords


@pytest.mark.requires_x64
@pytest.mark.parametrize("nelec", [2, 3, 6])
@pytest.mark.parametrize("track_matrix", [True, False], ids=["neural", "constant"])
def test_sparse_spinor_uses_custom_rule(fallbacks, nelec, track_matrix):
    """Per-electron spinors take the custom rule and match the dense oracle."""
    fn, coords = _neural_matrix_pair_factor(nelec, track_matrix=track_matrix)

    check_with_brute_force(
        fn,
        make_laplacian_input(coords, sparse_axis=0),
        rtol=1e-10,
        atol=1e-10,
    )
    assert fallbacks == []


@pytest.mark.requires_x64
def test_dense_spinor_falls_back(fallbacks):
    """Densely seeded spinors defer to the automatic Forward Laplacian."""
    fn, coords = _neural_matrix_pair_factor(3)

    check_with_brute_force(fn, coords, rtol=1e-10, atol=1e-10)
    assert fallbacks == ["pair factor spinor is not per-electron sparse"]


@pytest.mark.requires_x64
@pytest.mark.parametrize("conditioning", ["constant", "backbone"])
@pytest.mark.parametrize("det_dependent", [True, False])
def test_learned_dressing_custom_laplacian_matches_brute_force(
    fallbacks, conditioning, det_dependent
):
    """Every learned dressing takes the custom rule end to end."""
    nelec, ndets = 5, 2
    coords = jax.random.normal(jax.random.PRNGKey(3), (nelec, 4), dtype=jnp.float64)
    dressing = Dressing(
        config=DressingConfig(
            type="full",
            degree=4,
            conditioning=conditioning,
            orbital_dependent=True,
            det_dependent=det_dependent,
        ),
        ndets=ndets,
    )
    orbitals = jnp.ones((ndets, nelec, nelec), dtype=jnp.complex128)

    def inputs(x):
        z = x[:, :2] + 1j * x[:, 2:]
        z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
        h_one = jnp.tanh(x) + jnp.mean(jnp.tanh(x), axis=0)
        return z[:, 0], z[:, 1], h_one

    params = dressing.init(jax.random.PRNGKey(0), orbitals, *inputs(coords))
    params = jax.tree.map(
        lambda p: 0.05 * jax.random.normal(jax.random.PRNGKey(5), p.shape, p.dtype),
        params,
    )

    def dressed(x):
        return dressing.apply(params, orbitals, *inputs(x))[1, 2]  # type: ignore

    check_with_brute_force(
        dressed,
        make_laplacian_input(coords, sparse_axis=0),
        rtol=1e-9,
        atol=1e-10,
    )
    assert fallbacks == []


@pytest.mark.requires_x64
def test_mhpo_learned_dressing_kinetic_energy_matches_scan(fallbacks):
    """The Hall kinetic estimator agrees across Laplacian modes with the rule."""
    nelec, flux = 4, 9
    wf = MHPO(
        ndets=2,
        num_heads=2,
        heads_dim=8,
        num_layers=1,
        dressing=DressingConfig(type="full", degree=2, conditioning="backbone"),
    )
    wire(
        wf,
        nspins=(nelec, 0),
        monopole_strength=flux / 2,
        flux=flux,
        interaction_strength=1.0,
        radius=float(jnp.sqrt(flux / 2)),
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    theta = jnp.arccos(jax.random.uniform(keys[0], (3, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(keys[1], (3, nelec), minval=-jnp.pi, maxval=jnp.pi)
    electrons = jnp.stack([theta, phi], axis=-1)
    params = wf.init_params(HallData(electrons=electrons[0]), keys[2])
    params = jax.tree.map(
        lambda p: p + 0.05 * jax.random.normal(jax.random.PRNGKey(5), p.shape, p.dtype),
        params,
    )

    def kinetic(mode):
        estimator = SphericalKinetic(
            monopole_strength=flux / 2,
            radius=float(jnp.sqrt(flux / 2)),
            mode=mode,
            f_log_psi_from_spinor=wf.logpsi_from_spinor,
        )

        def single(x):
            stats, _ = estimator.evaluate_single_walker(
                params, HallData(electrons=x), {}, None, jax.random.PRNGKey(0)
            )
            return stats["energy:kinetic"]

        return jax.jit(jax.vmap(single))(electrons)

    np.testing.assert_allclose(
        kinetic(LaplacianMode.forward_laplacian),
        kinetic(LaplacianMode.scan),
        rtol=1e-8,
    )
    assert fallbacks == []
