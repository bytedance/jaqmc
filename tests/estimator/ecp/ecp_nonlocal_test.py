# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
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
from jaqmc.estimator.ecp.quadrature import Octahedron


def _constant_wf(flat_electrons):
    # Constant wavefunction: psi = 1 everywhere.
    return jnp.array(1.0), jnp.array(0.0)


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
