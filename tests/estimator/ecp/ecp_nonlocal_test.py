# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for ECP nonlocal integral Legendre polynomial indexing.

For a constant wavefunction (psi(r')/psi(r) = 1 everywhere), Legendre
polynomial orthogonality gives exact analytical results:

    integral P_0 dOmega = 4*pi    (P_0 = 1)
    integral P_l dOmega = 0       for all l > 0

This means only the l=0 (s-wave) semi-local channel should contribute
to the nonlocal energy. In PySCF's ECP format, the s-wave semi-local
coefficient is at index 1 (index 0 is local). A previous off-by-one bug
used P_1 instead of P_0 for this channel, zeroing out the contribution.
"""

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.estimator.ecp.nonlocal_integral import make_nonlocal_integral
from jaqmc.estimator.ecp.quadrature import Icosahedron, Octahedron


def _constant_wf(flat_electrons):
    # Constant wavefunction: psi = 1 everywhere.
    return jnp.array(1.0), jnp.array(0.0)


def _phase_and_logabs(value):
    return value / jnp.abs(value), jnp.log(jnp.abs(value))


def test_nonlocal_integral_legendre_indexing():
    """Only the l=0 (P_0) channel should contribute for a constant wf.

    With ecp_values = [0, V_s, 0, 0] (only s-wave nonzero), the nonlocal
    energy should equal V_s * (2*0+1)/(4pi) * integral(P_0 * 1) = V_s.

    If P_l indexing is off by one, the code would use P_1 (which integrates
    to 0) instead of P_0, giving zero nonlocal energy.
    """
    key = jax.random.key(42)
    electrons = jnp.array([[1.0, 0.0, 0.0]])
    atom = jnp.array([0.0, 0.0, 0.0])

    # atom_positions: nearest atom image for each electron
    atom_positions = atom[None, None, :]  # (1, 1, 3)

    v_s = 2.5
    # num_channels=4: indices [local, s-wave, p-wave, d-wave]
    # Only s-wave (index 1) is nonzero
    ecp_values = jnp.array([[[0.0, v_s, 0.0, 0.0]]])
    num_channels = 4
    quadrature = Octahedron(26)

    nonlocal_integral = make_nonlocal_integral(num_channels, quadrature)
    integrals = nonlocal_integral(_constant_wf, electrons, atom_positions, key)
    energy = jnp.sum(integrals * ecp_values[..., 1:])

    # (2*0+1)/(4pi) * integral(P_0 * 1, dOmega) = 1/(4pi) * 4pi = 1
    # So nonlocal energy = V_s * 1 = V_s
    np.testing.assert_allclose(float(energy), v_s, atol=1e-5)


def test_higher_channels_zero_for_constant_wf():
    """P-wave and d-wave channels should not contribute for a constant wf.

    With ecp_values = [0, 0, V_p, 0], the nonlocal energy should be zero
    because integral(P_1 * 1, dOmega) = 0 by orthogonality.
    """
    key = jax.random.key(99)
    electrons = jnp.array([[1.0, 0.0, 0.0]])
    atom = jnp.array([0.0, 0.0, 0.0])

    # atom_positions: nearest atom image for each electron
    atom_positions = atom[None, None, :]  # (1, 1, 3)

    # Only p-wave (index 2) is nonzero
    ecp_values = jnp.array([[[0.0, 0.0, 5.0, 0.0]]])
    num_channels = 4
    quadrature = Octahedron(26)

    nonlocal_integral = make_nonlocal_integral(num_channels, quadrature)
    integrals = nonlocal_integral(_constant_wf, electrons, atom_positions, key)
    energy = jnp.sum(integrals * ecp_values[..., 1:])

    np.testing.assert_allclose(float(energy), 0.0, atol=1e-5)


def test_nonlocal_integral_linear_wavefunction_oracle():
    r"""A linear wavefunction locks the P0/P1 signs and ``2l+1`` factors.

    For an electron at :math:`R\hat z` and
    :math:`\psi(\mathbf r)=1+c z`, spherical integration gives

    .. math::

        W_0 = \frac{1}{1+cR}, \qquad
        W_1 = \frac{cR}{1+cR}, \qquad
        W_2 = 0.
    """
    radius = 0.7
    coefficient = 0.4
    electrons = jnp.array([[0.0, 0.0, radius]])
    atom_positions = jnp.zeros((1, 1, 3))

    def linear_wf(displaced_electrons):
        value = 1.0 + coefficient * displaced_electrons[0, 2]
        return _phase_and_logabs(value)

    integral = make_nonlocal_integral(4, Octahedron(26))
    actual = integral(
        linear_wf,
        electrons,
        atom_positions,
        jax.random.key(123),
    )[0, 0]
    denominator = 1.0 + coefficient * radius
    expected = jnp.array([1.0 / denominator, coefficient * radius / denominator, 0.0])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_periodic_bloch_phase_matches_unwrapped_complex_oracle():
    """PBC wrapping plus Bloch phase reproduces the unwrapped complex ratio."""
    lattice = jnp.eye(3)
    twist = jnp.array([0.25, 0.125, 0.0])
    wavevector = 2 * jnp.pi * twist
    electrons = jnp.array([[0.15, 0.55, 0.5]])
    # The nearest image is outside the primary cell, so several quadrature
    # proposals cross the boundary and exercise the phase correction.
    atom_positions = jnp.array([[[-0.05, 0.5, 0.5]]])

    def plane_wave(electron_positions):
        value = jnp.exp(1j * jnp.dot(electron_positions[0], wavevector))
        return value, jnp.array(0.0)

    quadrature = Icosahedron(32)
    key = jax.random.key(91)
    unwrapped = make_nonlocal_integral(4, quadrature)(
        plane_wave, electrons, atom_positions, key
    )
    periodic = make_nonlocal_integral(4, quadrature, lattice=lattice, twist=twist)(
        plane_wave, electrons, atom_positions, key
    )

    np.testing.assert_allclose(periodic, unwrapped, atol=2e-6, rtol=2e-6)
    assert jnp.iscomplexobj(periodic)
