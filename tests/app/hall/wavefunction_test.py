# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hall wavefunctions and their building blocks."""

from typing import cast

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.wavefunction import _pair_factor
from jaqmc.app.hall.wavefunction._dressing import (
    PAIR_SPINOR_MATRIX,
    Dressing,
    DressingConfig,
)
from jaqmc.app.hall.wavefunction._jastrow import SphericalCoulombJastrow
from jaqmc.app.hall.wavefunction._pair_factor import log_pair_factor
from jaqmc.app.hall.wavefunction.free import Free
from jaqmc.app.hall.wavefunction.laughlin import Laughlin
from jaqmc.app.hall.wavefunction.mhpo import MHPO
from jaqmc.estimator.angular_momentum import SphericalAngularMomentum
from jaqmc.estimator.kinetic import SphericalKinetic
from jaqmc.geometry.sphere import cartesian_from_spinor, spinor_coordinates_from_angles
from jaqmc.laplacian import make_laplacian_input
from jaqmc.utils.wiring import wire
from tests.jax_version import requires_forward_laplacian
from tests.laplacian.helpers import check_with_brute_force


def _sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def _cartesian(electrons):
    return cartesian_from_spinor(*spinor_coordinates_from_angles(electrons))


def _free(nspins, flux):
    wf = Free()
    wire(wf, nspins=nspins, flux=flux, interaction_strength=1.0)
    return wf


def _laughlin(nspins, flux, excitation_lz=0):
    wf = Laughlin(flux_per_elec=1, excitation_lz=excitation_lz)
    wire(wf, nspins=nspins, flux=flux, interaction_strength=1.0)
    return wf


def _mhpo(
    nspins,
    flux,
    backbone="psiformer",
    orbital_readout="direct",
    dressing=DressingConfig(),
):
    wf = MHPO(
        ndets=1,
        backbone=backbone,
        num_heads=2,
        heads_dim=8,
        num_layers=1,
        orbital_readout=orbital_readout,
        dressing=dressing,
    )
    wire(
        wf,
        nspins=nspins,
        monopole_strength=flux / 2,
        flux=flux,
        interaction_strength=1.0,
        radius=float(jnp.sqrt(flux / 2)),
    )
    return wf


def _init(wf, nelec, seed=0):
    electrons = _sample(jax.random.PRNGKey(seed), 1, nelec)[0]
    params = wf.init_params(HallData(electrons=electrons), jax.random.PRNGKey(1))
    return params, electrons


def _logpsi_fn(wf, params):
    return lambda e: wf.evaluate(params, HallData(electrons=e))["logpsi"]


def _batch_eval(estimator, params, electrons):
    def single(x):
        stats, _ = estimator.evaluate_single_walker(
            params, HallData(electrons=x), {}, None, jax.random.PRNGKey(0)
        )
        return stats

    return jax.jit(jax.vmap(single))(electrons)


def _kinetic_near_pole(estimator, params, electrons, theta):
    """Local kinetic energies with electron 0 at ``theta`` over probe phis."""
    local_energies = []
    for phi in (0.0, 1.3, -2.1):
        near_pole = electrons.at[0].set(jnp.array([theta, phi]))
        stats, _ = estimator.evaluate_single_walker(
            params, HallData(electrons=near_pole), {}, None, jax.random.PRNGKey(0)
        )
        local_energies.append(stats["energy:kinetic"])
    return np.asarray(local_energies)


FULL_DRESSING = DressingConfig(type="full", degree=2)

WAVEFUNCTIONS = {
    "free": lambda: _free((3, 0), flux=2),
    "free-mixed-spins": lambda: _free((2, 1), flux=2),
    "laughlin": lambda: _laughlin((3, 0), flux=6),
    "mhpo-psiformer": lambda: _mhpo((3, 0), flux=4),
    "mhpo-lapnet": lambda: _mhpo((3, 0), flux=4, backbone="lapnet"),
    "mhpo-mixed-spins": lambda: _mhpo((2, 1), flux=4),
    "mhpo-factorized-psiformer": lambda: _mhpo(
        (3, 0), flux=4, orbital_readout="factorized"
    ),
    "mhpo-factorized-lapnet": lambda: _mhpo(
        (3, 0), flux=4, backbone="lapnet", orbital_readout="factorized"
    ),
    "mhpo-laughlin-dressing": lambda: _mhpo(
        (3, 0), flux=4, dressing=DressingConfig(type="laughlin", degree=2)
    ),
    "mhpo-full-dressing-constant": lambda: _mhpo(
        (3, 0),
        flux=4,
        backbone="lapnet",
        dressing=DressingConfig(type="full", degree=2, conditioning="constant"),
    ),
    "mhpo-full-dressing-backbone-factorized": lambda: _mhpo(
        (3, 0), flux=6, orbital_readout="factorized", dressing=FULL_DRESSING
    ),
}


@pytest.mark.parametrize("make_wf", WAVEFUNCTIONS.values(), ids=WAVEFUNCTIONS.keys())
class TestWavefunctionInvariants:
    """Properties every Hall wavefunction must satisfy on the sphere."""

    def test_same_spin_swap_flips_sign(self, make_wf):
        wf = make_wf()
        params, electrons = _init(wf, sum(wf.nspins))
        logpsi = _logpsi_fn(wf, params)

        swapped = electrons.at[0].set(electrons[1]).at[1].set(electrons[0])
        ratio = jnp.exp(logpsi(swapped) - logpsi(electrons))
        np.testing.assert_allclose(complex(ratio), -1.0, atol=1e-3)

    def test_phi_is_periodic(self, make_wf):
        wf = make_wf()
        params, electrons = _init(wf, sum(wf.nspins))
        logpsi = _logpsi_fn(wf, params)

        shifted = electrons.at[0, 1].add(2 * jnp.pi)
        np.testing.assert_allclose(
            complex(logpsi(shifted)), complex(logpsi(electrons)), atol=1e-4
        )

    def test_density_is_phi_independent_at_pole(self, make_wf):
        r"""At a pole, \|psi\|^2 must not depend on phi (subleading terms ~ theta^2)."""
        wf = make_wf()
        params, electrons = _init(wf, sum(wf.nspins))
        logpsi = _logpsi_fn(wf, params)

        re_logpsis = [
            float(jnp.real(logpsi(electrons.at[0].set(jnp.array([5e-4, phi])))))
            for phi in (0.0, 1.0, 2.5, -1.3)
        ]
        np.testing.assert_allclose(re_logpsis, re_logpsis[0], atol=1e-2)


class TestSphericalCoulombJastrow:
    @pytest.mark.x64_modes
    def test_parameters_are_float32(self, x64_mode):
        input_dtype = jnp.float64 if x64_mode else jnp.float32
        electrons = _sample(jax.random.PRNGKey(0), 1, 3)[0].astype(input_dtype)
        params = SphericalCoulombJastrow(
            nspins=(2, 1), interaction_strength=1.0, radius=1.0
        ).init(jax.random.PRNGKey(1), _cartesian(electrons))

        assert all(param.dtype == jnp.float32 for param in jax.tree.leaves(params))

    @pytest.mark.parametrize(
        "nspins", [(3, 0), (2, 1), (1, 2)], ids=["all-up", "two-up", "two-down"]
    )
    def test_symmetric_under_same_spin_swap(self, nspins):
        jastrow = SphericalCoulombJastrow(
            nspins=nspins, interaction_strength=1.0, radius=1.0
        )
        electrons = _sample(jax.random.PRNGKey(7), 1, 3)[0]
        params = jastrow.init(jax.random.PRNGKey(1), _cartesian(electrons))
        # Swap the two electrons sharing the majority spin channel.
        i, j = (0, 1) if nspins[0] >= 2 else (1, 2)
        swapped = electrons.at[i].set(electrons[j]).at[j].set(electrons[i])

        original = cast(jax.Array, jastrow.apply(params, _cartesian(electrons)))
        swapped_out = cast(jax.Array, jastrow.apply(params, _cartesian(swapped)))
        np.testing.assert_allclose(swapped_out, original, atol=1e-5)
        assert jnp.isfinite(original)

    def test_interaction_strength_scales_jastrow(self):
        """The exact 2D Coulomb cusp scales with interaction strength."""
        cartesian = _cartesian(_sample(jax.random.PRNGKey(9), 1, 3)[0])

        def jastrow(strength):
            return SphericalCoulombJastrow(
                nspins=(2, 1), interaction_strength=strength, radius=1.0
            )

        params = jastrow(1.0).init(jax.random.PRNGKey(1), cartesian)
        np.testing.assert_allclose(
            jastrow(2.5).apply(params, cartesian),
            2.5 * jastrow(1.0).apply(params, cartesian),
            rtol=1e-6,
        )


class TestFree:
    @requires_forward_laplacian
    def test_filled_lll_kinetic_energy(self):
        """Filling the LLL gives exactly 1/2 kinetic energy per electron."""
        flux = 4
        wf = _free((3, 0), flux=flux)
        params, _ = _init(wf, 3)
        estimator = SphericalKinetic(
            monopole_strength=flux / 2, f_log_psi_from_spinor=wf.logpsi_from_spinor
        )
        stats = _batch_eval(estimator, params, _sample(jax.random.PRNGKey(42), 4, 3))
        np.testing.assert_allclose(stats["energy:kinetic"], 1.5, atol=1e-3)

    def test_overflow_into_next_landau_level(self):
        """More electrons than LLL orbitals fill the next Landau level."""
        wf = _free((2, 0), flux=0)
        params, electrons = _init(wf, 2)
        assert jnp.isfinite(_logpsi_fn(wf, params)(electrons))


LAUGHLIN_STATES = pytest.mark.parametrize(
    "nelec,Q,excitation_lz",
    [(3, 3, 0), (4, 4.5, 0), (4, 5, 2), (6, 7, 1)],
)


class TestLaughlin:
    def test_rejects_unsupported_filling(self):
        wf = _laughlin((2, 0), flux=6)
        with pytest.raises(ValueError, match="Unsupported Laughlin filling"):
            _init(wf, 2)

    @LAUGHLIN_STATES
    @requires_forward_laplacian
    def test_kinetic_energy_is_exact(self, nelec, Q, excitation_lz):
        wf = _laughlin((nelec, 0), flux=int(2 * Q), excitation_lz=excitation_lz)
        params, _ = _init(wf, nelec)
        estimator = SphericalKinetic(
            monopole_strength=Q, f_log_psi_from_spinor=wf.logpsi_from_spinor
        )
        electrons = _sample(jax.random.PRNGKey(1898), 2, nelec)
        stats = _batch_eval(estimator, params, electrons)
        np.testing.assert_allclose(stats["energy:kinetic"], nelec / 2, atol=1e-3)

    @LAUGHLIN_STATES
    def test_angular_momentum_eigenstate(self, nelec, Q, excitation_lz):
        wf = _laughlin((nelec, 0), flux=int(2 * Q), excitation_lz=excitation_lz)
        params, _ = _init(wf, nelec)
        estimator = SphericalAngularMomentum(
            f_log_psi_from_spinor=wf.logpsi_from_spinor
        )
        electrons = _sample(jax.random.PRNGKey(1898), 2, nelec)
        stats = _batch_eval(estimator, params, electrons)

        np.testing.assert_allclose(
            stats["angular_momentum_z"], excitation_lz, atol=1e-3
        )
        np.testing.assert_allclose(
            stats["angular_momentum_z_square"], excitation_lz**2, atol=1e-3
        )
        if excitation_lz == 0:
            # Only ground states are L^2 eigenstates; excitations carry extra
            # angular momentum in the quasihole/quasiparticle coordinates.
            np.testing.assert_allclose(stats["angular_momentum_square"], 0, atol=1e-3)

    @requires_forward_laplacian
    def test_kinetic_stable_near_pole(self):
        """Stereographic kinetic stays on the exact Laughlin value near a pole.

        Float32 measurements for ``nelec=3``, ``Q=3`` keep
        ``max|E - 1.5|`` around ``1e-6`` down to ``theta=1e-7``; the
        tolerances below leave a few times headroom on that path.
        """
        nelec, Q = 3, 3.0
        wf = _laughlin((nelec, 0), flux=int(2 * Q))
        params, _ = _init(wf, nelec)
        electrons = _sample(jax.random.PRNGKey(1898), 1, nelec)[0]
        estimator = SphericalKinetic(
            monopole_strength=Q, f_log_psi_from_spinor=wf.logpsi_from_spinor
        )

        for theta in (1e-3, 1e-5, 1e-7):
            local_energies = _kinetic_near_pole(estimator, params, electrons, theta)
            assert np.all(np.isfinite(local_energies))
            np.testing.assert_allclose(local_energies, nelec / 2, rtol=0, atol=5e-6)
            np.testing.assert_allclose(
                local_energies, local_energies[0], rtol=0, atol=5e-6
            )


class TestMHPO:
    @requires_forward_laplacian
    def test_kinetic_regular_near_pole(self):
        """MHPO local kinetic stays finite and phi-stable near a pole.

        Exactness of the stereographic estimator is covered by Laughlin;
        this only checks that the neural ansatz remains regular enough
        for the kinetic to settle near the pole.
        """
        flux = 3
        wf = _mhpo((2, 0), flux=flux)
        params, electrons = _init(wf, 2, seed=81)
        estimator = SphericalKinetic(
            monopole_strength=flux / 2, f_log_psi_from_spinor=wf.logpsi_from_spinor
        )

        control = _kinetic_near_pole(estimator, params, electrons, 1e-3)
        assert np.all(np.isfinite(control))
        assert np.max(np.abs(control)) < 15

        for theta in (1e-5, 1e-7):
            local_energies = _kinetic_near_pole(estimator, params, electrons, theta)
            assert np.all(np.isfinite(local_energies))
            np.testing.assert_allclose(local_energies, control, rtol=0, atol=5e-2)
            np.testing.assert_allclose(
                local_energies, local_energies[0], rtol=0, atol=1e-3
            )

    def test_learned_dressing_has_finite_gradients(self):
        wf = _mhpo((3, 0), flux=6, orbital_readout="factorized", dressing=FULL_DRESSING)
        params, electrons = _init(wf, 3)
        data = HallData(electrons=electrons)

        grads = jax.grad(lambda p: jnp.real(wf.logpsi(p, data)))(params)
        leaves = jax.tree.leaves(grads["params"]["dressing_layer"])
        assert leaves
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


@pytest.fixture
def pair_factor_fallbacks(monkeypatch):
    """Record every fallback the pair-factor Laplacian rule raises.

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


@requires_forward_laplacian
@pytest.mark.requires_x64
class TestPairFactorLaplacianRule:
    """The custom Forward Laplacian rule for pair factors matches brute force."""

    @pytest.mark.parametrize("nelec", [2, 3, 6])
    @pytest.mark.parametrize("track_matrix", [True, False], ids=["neural", "constant"])
    def test_sparse_spinor_uses_custom_rule(
        self, pair_factor_fallbacks, nelec, track_matrix
    ):
        fn, coords = _neural_matrix_pair_factor(nelec, track_matrix=track_matrix)

        check_with_brute_force(
            fn, make_laplacian_input(coords, sparse_axis=0), rtol=1e-10, atol=1e-10
        )
        assert pair_factor_fallbacks == []

    def test_dense_spinor_falls_back(self, pair_factor_fallbacks):
        fn, coords = _neural_matrix_pair_factor(3)

        check_with_brute_force(fn, coords, rtol=1e-10, atol=1e-10)
        assert pair_factor_fallbacks == [
            "pair factor spinor is not per-electron sparse"
        ]

    @pytest.mark.parametrize("conditioning", ["constant", "backbone"])
    @pytest.mark.parametrize("det_dependent", [True, False])
    def test_learned_dressing_uses_custom_rule(
        self, pair_factor_fallbacks, conditioning, det_dependent
    ):
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
            return cast(jax.Array, dressing.apply(params, orbitals, *inputs(x)))[1, 2]

        check_with_brute_force(
            dressed,
            make_laplacian_input(coords, sparse_axis=0),
            rtol=1e-9,
            atol=1e-10,
        )
        assert pair_factor_fallbacks == []
