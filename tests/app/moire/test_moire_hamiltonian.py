# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.hamiltonian import (
    MoirePotential,
    MoireSOC,
    _moire_first_shell_vectors,
    _moire_valley_momenta,
    compute_operator_derivative_ratio,
    compute_operator_ratio,
)


def test_compute_operator_ratio_matches_multilayer_determinant_oracle():
    nlayers, ndets, nelec = 3, 2, 2
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    phi = 0.15 * (
        jax.random.normal(keys[0], (nlayers, ndets, nelec, nelec))
        + 1j * jax.random.normal(keys[1], (nlayers, ndets, nelec, nelec))
    )
    phi = phi.at[0].add(jnp.stack([jnp.eye(nelec), 1.2 * jnp.eye(nelec)]))
    chi = jnp.asarray([[1.0 + 0.2j, -0.3 + 0.1j, 0.2j], [0.8, 0.1j, -0.2]])
    operators = jax.random.normal(
        keys[2], (2, nelec, nlayers, nlayers)
    ) + 1j * jax.random.normal(keys[3], (2, nelec, nlayers, nlayers))
    base_orbitals = jnp.einsum("il,ldij->dij", chi, phi)
    operator_orbitals = jnp.einsum("il,ailm,mdij->adij", chi, operators, phi)

    actual = compute_operator_ratio(base_orbitals, operator_orbitals)
    psi = jnp.sum(jax.vmap(jnp.linalg.det)(base_orbitals))
    expected = []
    for transformed in operator_orbitals:
        per_electron = []
        for electron_idx in range(nelec):
            numerator = sum(
                jnp.linalg.det(
                    base_orbitals[det_idx]
                    .at[electron_idx]
                    .set(transformed[det_idx, electron_idx])
                )
                for det_idx in range(ndets)
            )
            per_electron.append(numerator / psi)
        expected.append(jnp.stack(per_electron))

    np.testing.assert_allclose(actual, jnp.stack(expected), rtol=1e-6, atol=1e-6)


def test_compute_operator_derivative_ratio_matches_direct_autodiff_oracle():
    ndets, nelec, ndim, noperators = 2, 2, 2, 2
    positions = jnp.asarray([[0.2, -0.1], [0.4, 0.3]])
    data = MoireData(positions=positions, spin_coords=jnp.asarray([0.3, -0.5]))
    base_const = jnp.asarray(
        [
            [[1.2 + 0.1j, 0.2 - 0.1j], [0.1 + 0.3j, 0.9 + 0.2j]],
            [[0.8 - 0.2j, -0.1 + 0.2j], [0.3 + 0.1j, 1.1 - 0.1j]],
        ]
    )
    base_grid = jnp.arange(8, dtype=positions.dtype).reshape(ndets, nelec, nelec) + 1
    operator_grid = jnp.arange(16, dtype=positions.dtype).reshape(
        noperators, ndets, nelec, nelec
    )
    operator_const = (
        0.2 + 0.03 * operator_grid + 1j * (-0.1 + 0.02 * operator_grid[..., ::-1])
    )

    def operator_orbitals_fn(params, walker_data):
        del params
        spatial = walker_data.positions
        base = (
            base_const
            + spatial[None, :, 0, None] * (0.01 + 0.02j) * base_grid
            + spatial[None, :, 1, None] * (-0.015 + 0.01j) * base_grid[..., ::-1]
            + jnp.sum(spatial[:, 0]) * (0.004 - 0.003j) * base_grid
        )
        transformed = (
            operator_const
            + spatial[None, None, :, 0, None] * (0.011 - 0.007j) * (operator_grid + 1)
            + spatial[None, None, :, 1, None]
            * (-0.009 + 0.013j)
            * (operator_grid[..., ::-1] + 1)
            + jnp.sum(spatial[:, 1]) * (0.003 + 0.002j) * (operator_grid + 1)
        )
        return base, transformed

    actual = compute_operator_derivative_ratio(
        {}, data, operator_orbitals_fn=operator_orbitals_fn
    )
    base_orbitals, _ = operator_orbitals_fn({}, data)
    psi = jnp.sum(jax.vmap(jnp.linalg.det)(base_orbitals))
    expected = []
    for operator_idx in range(noperators):
        per_electron = []
        for electron_idx in range(nelec):

            def numerator(spatial_flat):
                walker_data = data.merge(
                    {"positions": spatial_flat.reshape(nelec, ndim)}
                )
                base, transformed = operator_orbitals_fn({}, walker_data)
                exchanged = jnp.stack(
                    [
                        base[det_idx]
                        .at[electron_idx]
                        .set(transformed[operator_idx, det_idx, electron_idx])
                        for det_idx in range(ndets)
                    ]
                )
                return jnp.sum(jax.vmap(jnp.linalg.det)(exchanged))

            derivative = jax.jacfwd(numerator)(positions.reshape(-1)).reshape(
                nelec, ndim
            )
            per_electron.append(derivative[electron_idx] / psi)
        expected.append(jnp.stack(per_electron))

    np.testing.assert_allclose(actual, jnp.stack(expected), rtol=1e-6, atol=1e-6)


def test_moire_potential_matches_pauli_decomposition_for_both_physical_spins():
    lattice = jnp.asarray([[jnp.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]])
    data = MoireData(
        positions=jnp.asarray([[0.1, 0.2], [0.23, -0.17]]),
        spin_coords=jnp.asarray([0.3, 1.1]),
    )
    grid = jnp.arange(16, dtype=data.positions.dtype).reshape(2, 2, 2, 2)
    phi = 0.04 * (grid + 1) + 0.03j * (grid[..., ::-1] + 1)
    phi = phi.at[0].add(jnp.stack([jnp.eye(2), 1.2 * jnp.eye(2)]))

    def layer_components(params, walker_data):
        del params
        chi = jnp.stack(
            [
                jnp.exp(1j * walker_data.spin_coords),
                jnp.exp(-1j * walker_data.spin_coords),
            ],
            axis=-1,
        )
        return phi, chi, jnp.einsum("il,ldij->dij", chi, phi)

    estimator = MoirePotential(
        f_layer_components=layer_components,
        primitive_lattice=lattice,
        nspins=(1, 1),
        v1_mev=0.7,
        phi1_rad=0.31,
        omega1_mev=-0.4,
    )
    stats, _ = estimator.evaluate_single_walker(
        {}, data, {}, None, jax.random.PRNGKey(1)
    )

    layer_phi, chi, phi0 = layer_components({}, data)
    pauli = jnp.asarray(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1j], [1j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ]
    )
    pauli_orbitals = jnp.einsum("il,alm,mdij->adij", chi, pauli, layer_phi)
    spin_ratio = compute_operator_ratio(phi0, pauli_orbitals)
    # This full-matrix integration check covers the layer-potential algebra and
    # Pauli/tunneling contraction.
    reciprocal = _moire_first_shell_vectors(lattice)
    projection = data.positions @ reciprocal[0:5:2].T
    v_bottom = -1.4 * jnp.sum(jnp.cos(projection + 0.31), axis=-1)
    v_top = -1.4 * jnp.sum(jnp.cos(projection - 0.31), axis=-1)
    tunnel = -0.4 * (
        1.0
        + jnp.exp(1j * (data.positions @ reciprocal[1]))
        + jnp.exp(1j * (data.positions @ reciprocal[2]))
    )
    eta = jnp.asarray([1.0, -1.0])
    expected = jnp.sum(
        (v_bottom + v_top) / 2.0
        + tunnel.real * spin_ratio[0]
        - eta * tunnel.imag * spin_ratio[1]
        + (v_bottom - v_top) / 2.0 * spin_ratio[2]
    )

    np.testing.assert_allclose(
        stats["energy:moire_potential"], expected, rtol=1e-6, atol=1e-6
    )


def test_moire_soc_matches_k_sum_and_difference_formula():
    lattice = jnp.asarray([[1.13, 0.17], [0.29, 0.91]])
    positions = jnp.asarray([[0.12, -0.07], [0.31, 0.26]])
    data = MoireData(positions=positions, spin_coords=jnp.asarray([0.2, -0.4]))
    grid = jnp.arange(16, dtype=positions.dtype).reshape(2, 2, 2, 2)
    phi_const = 0.025 * (grid + 1) + 0.018j * (grid[..., ::-1] + 1)
    phi_const = phi_const.at[0].add(jnp.stack([jnp.eye(2), 1.1 * jnp.eye(2)]))

    def layer_components(params, walker_data):
        del params
        spatial = walker_data.positions
        phi = (
            phi_const
            + spatial[None, None, :, 0, None] * (0.009 - 0.006j) * (grid + 1)
            + spatial[None, None, :, 1, None]
            * (-0.005 + 0.008j)
            * (grid[..., ::-1] + 1)
        )
        chi = jnp.stack(
            [
                jnp.exp(1j * walker_data.spin_coords),
                jnp.exp(-1j * walker_data.spin_coords),
            ],
            axis=-1,
        )
        return phi, chi, jnp.einsum("il,ldij->dij", chi, phi)

    estimator = MoireSOC(
        f_layer_components=layer_components,
        primitive_lattice=lattice,
        nspins=(1, 1),
    )
    stats, _ = estimator.evaluate_single_walker(
        {}, data, {}, None, jax.random.PRNGKey(2)
    )

    def amplitude(spatial_flat):
        walker_data = data.merge({"positions": spatial_flat.reshape(2, 2)})
        _, _, phi0 = layer_components({}, walker_data)
        return jnp.sum(jax.vmap(jnp.linalg.det)(phi0))

    spatial_flat = positions.reshape(-1)
    psi = amplitude(spatial_flat)
    drift = jax.jacfwd(amplitude)(spatial_flat).reshape(2, 2) / psi
    sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]])
    phi, chi, phi0 = layer_components({}, data)
    z_orbitals = jnp.einsum("il,lm,mdij->dij", chi, sigma_z, phi)
    z_ratio = compute_operator_ratio(phi0, z_orbitals)
    spin_drift_entries: list[jnp.ndarray] = []
    for electron_idx in range(2):

        def spin_numerator(flat):
            walker_data = data.merge({"positions": flat.reshape(2, 2)})
            phi, chi, phi0 = layer_components({}, walker_data)
            transformed = jnp.einsum("il,lm,mdij->dij", chi, sigma_z, phi)
            exchanged = jnp.stack(
                [
                    phi0[det_idx]
                    .at[electron_idx]
                    .set(transformed[det_idx, electron_idx])
                    for det_idx in range(2)
                ]
            )
            return jnp.sum(jax.vmap(jnp.linalg.det)(exchanged))

        derivative = jax.jacfwd(spin_numerator)(spatial_flat).reshape(2, 2)
        spin_drift_entries.append(derivative[electron_idx] / psi)
    spin_drift = jnp.stack(spin_drift_entries)

    k_plus, k_minus = _moire_valley_momenta(lattice)
    eta = jnp.asarray([1.0, -1.0])[:, None]
    drift_energy = 0.5j * jnp.sum(
        eta
        * (
            drift * (k_plus + k_minus)[None, :]
            + spin_drift * (k_plus - k_minus)[None, :]
        )
    )
    q_sum = jnp.sum(k_plus**2) + jnp.sum(k_minus**2)
    q_diff = jnp.sum(k_plus**2) - jnp.sum(k_minus**2)
    assert not np.isclose(q_diff, 0.0)
    # Unlike the drift, Q has no physical-spin eta factor.
    q_energy = jnp.sum(q_sum / 4.0 + q_diff / 4.0 * z_ratio)
    expected = drift_energy + q_energy

    np.testing.assert_allclose(stats["energy:soc"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("estimator_cls", [MoirePotential, MoireSOC])
def test_bilayer_estimators_reject_three_layer_components(estimator_cls):
    def layer_components(params, data):
        del params
        phi = jnp.ones((3, 1, 2, 2), dtype=jnp.complex64)
        chi = jnp.ones((2, 3), dtype=jnp.complex64)
        return phi, chi, jnp.einsum("il,ldij->dij", chi, phi)

    estimator = estimator_cls(
        f_layer_components=layer_components,
        primitive_lattice=jnp.eye(2),
        nspins=(2, 0),
    )
    data = MoireData(positions=jnp.zeros((2, 2)), spin_coords=jnp.zeros((2,)))

    with pytest.raises(ValueError, match="exactly two layers"):
        estimator.evaluate_single_walker({}, data, {}, None, jax.random.PRNGKey(3))


def test_default_hexagonal_lattice_matches_expected_moire_wavevectors():
    lattice = jnp.asarray([[jnp.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]])
    first_shell = np.asarray(_moire_first_shell_vectors(lattice))
    k_plus, k_minus = _moire_valley_momenta(lattice)

    expected_first_shell = (
        np.asarray(
            [[np.cos(np.pi / 3.0 * i), np.sin(np.pi / 3.0 * i)] for i in range(6)]
        )
        * 4.0
        * np.pi
        / np.sqrt(3.0)
    )
    expected_k_plus = 4.0 * np.pi / np.sqrt(3.0) * np.asarray([0.5, np.sqrt(3.0) / 6.0])
    expected_k_minus = (
        4.0 * np.pi / np.sqrt(3.0) * np.asarray([0.5, -np.sqrt(3.0) / 6.0])
    )

    np.testing.assert_allclose(first_shell, expected_first_shell, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(k_plus), expected_k_plus, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(k_minus), expected_k_minus, atol=1e-6, rtol=1e-6
    )
