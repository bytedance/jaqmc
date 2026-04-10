# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Tests for ECP potential evaluation and nearest-core selection.

Tests ``ECPRadial`` using synthetic ECP coefficients in PySCF
format against the analytical formula:

.. math::

    V_l(r) = \sum_k c_k \, r^{n_k - 2} \exp(-\alpha_k \, r^2)

Tests ``_select_nearest_cores`` for correct distance-based filtering.
"""

from typing import ClassVar

import numpy as np
from jax import numpy as jnp

from jaqmc.estimator.ecp.estimator import ECPRadial, _select_nearest_cores


def _compute_ecp_geometry(electrons, atoms, atom_symbols, ecp_coefficients):
    """Compute electron-atom geometry and ECP radial potential for tests.

    Returns:
        Tuple of (ecp_atom_indices, ecp_radial, r_ea_vectors, r_ea_distances).
        ecp_radial is ECPRadial or None if no ECP atoms.
    """
    ecp_atom_indices = [
        i for i, sym in enumerate(atom_symbols) if sym in ecp_coefficients
    ]
    if not ecp_atom_indices:
        n_elec = electrons.shape[0]
        return [], None, jnp.zeros((n_elec, 0, 3)), jnp.zeros((n_elec, 0))

    num_channels = max(len(ecp_coefficients[sym][1]) for sym in ecp_coefficients)
    ecp_radial = ECPRadial.from_pyscf(
        ecp_coefficients, atom_symbols, ecp_atom_indices, num_channels
    )
    ecp_atoms = atoms[jnp.array(ecp_atom_indices)]
    r_ea_vectors = electrons[:, None, :] - ecp_atoms[None, :, :]
    r_ea_distances = jnp.linalg.norm(r_ea_vectors, axis=-1)
    return ecp_atom_indices, ecp_radial, r_ea_vectors, r_ea_distances


def _make_ecp_coefficients(channels):
    """Build a single-element ECP coefficient dict in PySCF format.

    PySCF uses l=-1 for the local channel and l=0,1,... for semi-local.
    The first entry in ``channels`` becomes the local (l=-1) channel;
    subsequent entries become l=0, l=1, etc.

    Args:
        channels: List of lists, one per angular momentum channel.
            Each channel is a list of (power_idx, alpha, c) tuples.

    Returns:
        ECP coefficient dict keyed by "X".
    """
    num_channels = len(channels)
    pyscf_channels = []
    for l_idx, terms in enumerate(channels):
        pyscf_l = l_idx - 1  # 0 → -1 (local), 1 → 0 (s), 2 → 1 (p), ...
        # Group terms by power_idx
        max_power = max((t[0] for t in terms), default=-1) + 1
        radial = [[] for _ in range(max_power)]
        for power_idx, alpha, c in terms:
            radial[power_idx].append([alpha, c])
        pyscf_channels.append([pyscf_l, radial])
    return {"X": [num_channels, pyscf_channels]}


class TestEvaluateEcpPotential:
    def test_single_channel(self):
        """Single atom, single electron, one channel: V_0(r) = 3 exp(-2r^2)."""
        electrons = jnp.array([[1.0, 0.0, 0.0]])
        atoms = jnp.array([[0.0, 0.0, 0.0]])
        atom_symbols = ["X"]
        # Channel 0: power_idx=2 -> r^(2-2) = 1, so V_0(r) = 3 * exp(-2r^2)
        ecp_coefficients = _make_ecp_coefficients([[(2, 2.0, 3.0)]])

        _, ecp_radial, r_ea_vectors, r_ea_distances = _compute_ecp_geometry(
            electrons, atoms, atom_symbols, ecp_coefficients
        )
        ecp_values = ecp_radial(r_ea_distances)

        assert ecp_values.shape == (1, 1, 1)
        assert r_ea_vectors.shape == (1, 1, 3)
        assert r_ea_distances.shape == (1, 1)

        expected = 3.0 * jnp.exp(-2.0)
        np.testing.assert_allclose(
            float(ecp_values[0, 0, 0]), float(expected), atol=1e-6
        )
        np.testing.assert_allclose(float(r_ea_distances[0, 0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(r_ea_vectors[0, 0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_two_channels(self):
        """Two channels with different radial dependence."""
        electrons = jnp.array([[2.0, 0.0, 0.0]])
        atoms = jnp.array([[0.0, 0.0, 0.0]])
        atom_symbols = ["X"]
        # Channel 0: power_idx=2 -> V_0 = 3 * exp(-2r^2)
        # Channel 1: power_idx=2 -> V_1 = 1.5 * exp(-r^2)
        ecp_coefficients = _make_ecp_coefficients(
            [
                [(2, 2.0, 3.0)],
                [(2, 1.0, 1.5)],
            ]
        )

        _, ecp_radial, _, r_ea_distances = _compute_ecp_geometry(
            electrons, atoms, atom_symbols, ecp_coefficients
        )
        ecp_values = ecp_radial(r_ea_distances)

        assert ecp_values.shape == (1, 1, 2)
        r = 2.0
        np.testing.assert_allclose(
            float(ecp_values[0, 0, 0]), 3.0 * np.exp(-2.0 * r**2), atol=1e-6
        )
        np.testing.assert_allclose(
            float(ecp_values[0, 0, 1]), 1.5 * np.exp(-1.0 * r**2), atol=1e-6
        )

    def test_multiple_terms(self):
        """Multiple terms in a single channel sum together."""
        electrons = jnp.array([[1.0, 0.0, 0.0]])
        atoms = jnp.array([[0.0, 0.0, 0.0]])
        atom_symbols = ["X"]
        # Channel 0: two terms at power_idx=2
        # V_0 = 2 * exp(-r^2) + 5 * exp(-3r^2)
        ecp_coefficients = _make_ecp_coefficients([[(2, 1.0, 2.0), (2, 3.0, 5.0)]])

        _, ecp_radial, _, r_ea_distances = _compute_ecp_geometry(
            electrons, atoms, atom_symbols, ecp_coefficients
        )
        ecp_values = ecp_radial(r_ea_distances)

        expected = 2.0 * np.exp(-1.0) + 5.0 * np.exp(-3.0)
        np.testing.assert_allclose(float(ecp_values[0, 0, 0]), expected, atol=1e-6)

    def test_no_ecp_atoms(self):
        """Geometry helper returns empty arrays when no atoms have ECP."""
        electrons = jnp.array([[1.0, 0.0, 0.0]])
        atoms = jnp.array([[0.0, 0.0, 0.0]])
        atom_symbols = ["H"]
        # "H" is not in the ECP dict
        ecp_coefficients = _make_ecp_coefficients([[(2, 1.0, 1.0)]])

        result = _compute_ecp_geometry(electrons, atoms, atom_symbols, ecp_coefficients)
        ecp_atom_indices, ecp_radial, r_ea_vectors, r_ea_distances = result

        assert ecp_atom_indices == []
        assert ecp_radial is None
        assert r_ea_vectors.shape == (1, 0, 3)
        assert r_ea_distances.shape == (1, 0)

    def test_multiple_electrons(self):
        """Multiple electrons at different distances from one atom."""
        electrons = jnp.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        atoms = jnp.array([[0.0, 0.0, 0.0]])
        atom_symbols = ["X"]
        ecp_coefficients = _make_ecp_coefficients([[(2, 1.0, 1.0)]])

        _, ecp_radial, _, r_ea_distances = _compute_ecp_geometry(
            electrons, atoms, atom_symbols, ecp_coefficients
        )
        ecp_values = ecp_radial(r_ea_distances)

        assert ecp_values.shape == (2, 1, 1)
        np.testing.assert_allclose(float(r_ea_distances[0, 0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(r_ea_distances[1, 0]), 3.0, atol=1e-6)
        np.testing.assert_allclose(float(ecp_values[0, 0, 0]), np.exp(-1.0), atol=1e-6)
        np.testing.assert_allclose(float(ecp_values[1, 0, 0]), np.exp(-9.0), atol=1e-6)

    def test_mixed_atoms(self):
        """Only atoms present in the ECP dict are included in geometry."""
        electrons = jnp.array([[1.0, 0.0, 0.0]])
        atoms = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        atom_symbols = ["X", "H"]
        # Only "X" has ECP; "H" does not
        ecp_coefficients = _make_ecp_coefficients([[(2, 1.0, 1.0)]])

        ecp_atom_indices, ecp_radial, _, r_ea_distances = _compute_ecp_geometry(
            electrons, atoms, atom_symbols, ecp_coefficients
        )
        ecp_values = ecp_radial(r_ea_distances)

        # Only 1 ECP atom (the "X" at origin)
        assert ecp_atom_indices == [0]
        assert ecp_values.shape == (1, 1, 1)
        np.testing.assert_allclose(float(r_ea_distances[0, 0]), 1.0, atol=1e-6)


class TestSelectNearestCores:
    def test_no_truncation(self):
        """When max_core >= n_ecp_atoms, arrays are returned unchanged."""
        ecp_values = jnp.array([[[1.0], [2.0]]])
        r_ea_vectors = jnp.ones((1, 2, 3))
        r_ea_distances = jnp.array([[1.0, 2.0]])

        sel_v, _sel_vec, sel_d = _select_nearest_cores(
            ecp_values, r_ea_vectors, r_ea_distances, max_core=5
        )

        np.testing.assert_array_equal(sel_v, ecp_values)
        np.testing.assert_array_equal(sel_d, r_ea_distances)

    def test_truncation(self):
        """Selects the closest max_core atoms per electron."""
        # 2 electrons, 3 atoms
        ecp_values = jnp.array(
            [
                [[10.0], [20.0], [30.0]],
                [[40.0], [50.0], [60.0]],
            ]
        )
        r_ea_vectors = jnp.zeros((2, 3, 3))
        r_ea_distances = jnp.array(
            [
                [1.0, 3.0, 2.0],  # electron 0: nearest are atoms 0, 2
                [3.0, 1.0, 2.0],  # electron 1: nearest are atoms 1, 2
            ]
        )

        sel_v, _, sel_d = _select_nearest_cores(
            ecp_values, r_ea_vectors, r_ea_distances, max_core=2
        )

        assert sel_v.shape == (2, 2, 1)
        assert sel_d.shape == (2, 2)

        # Electron 0: atoms 0 (d=1.0) and 2 (d=2.0) selected, sorted by distance
        np.testing.assert_allclose(sel_d[0], [1.0, 2.0])
        np.testing.assert_allclose(sel_v[0, :, 0], [10.0, 30.0])

        # Electron 1: atoms 1 (d=1.0) and 2 (d=2.0) selected
        np.testing.assert_allclose(sel_d[1], [1.0, 2.0])
        np.testing.assert_allclose(sel_v[1, :, 0], [50.0, 60.0])

    def test_single_core(self):
        """max_core=1 selects only the closest atom per electron."""
        ecp_values = jnp.array([[[1.0], [2.0], [3.0]]])
        r_ea_vectors = jnp.zeros((1, 3, 3))
        r_ea_distances = jnp.array([[5.0, 1.0, 3.0]])

        sel_v, _, sel_d = _select_nearest_cores(
            ecp_values, r_ea_vectors, r_ea_distances, max_core=1
        )

        assert sel_d.shape == (1, 1)
        np.testing.assert_allclose(float(sel_d[0, 0]), 1.0)
        np.testing.assert_allclose(float(sel_v[0, 0, 0]), 2.0)


class TestPySCFChannelMapping:
    """Verify that from_pyscf maps l=-1 to channel 0 and l=0,1,... to 1,2,..."""

    # Synthetic PySCF-format ECP: two channels with distinct, known terms.
    #   l=-1 (local):  one term at power_idx=1 → power=-1, alpha=3.0, c=7.0
    #   l= 0 (s-wave): one term at power_idx=2 → power= 0, alpha=1.0, c=5.0
    MOCK_ECP: ClassVar = {
        "X": [
            2,  # n_core_electrons (unused by from_pyscf)
            [
                # local: power_idx=0 empty, power_idx=1 has one term
                [-1, [[], [[3.0, 7.0]]]],
                # s-wave: power_idx=0,1 empty, power_idx=2 has one term
                [0, [[], [], [[1.0, 5.0]]]],
            ],
        ]
    }

    def test_local_channel_at_index_zero(self):
        """l=-1 (local) data lands in channel 0."""
        ecp = ECPRadial.from_pyscf(self.MOCK_ECP, ["X"], [0], num_channels=2)

        np.testing.assert_allclose(ecp.alphas[0, 0, 0], 3.0)
        np.testing.assert_allclose(ecp.coeffs[0, 0, 0], 7.0)
        np.testing.assert_allclose(ecp.powers[0, 0, 0], -1.0)  # power_idx=1 → 1-2 = -1

    def test_swave_channel_at_index_one(self):
        """l=0 (s-wave) data lands in channel 1."""
        ecp = ECPRadial.from_pyscf(self.MOCK_ECP, ["X"], [0], num_channels=2)

        np.testing.assert_allclose(ecp.alphas[0, 1, 0], 1.0)
        np.testing.assert_allclose(ecp.coeffs[0, 1, 0], 5.0)
        np.testing.assert_allclose(ecp.powers[0, 1, 0], 0.0)  # power_idx=2 → 2-2 = 0
