# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for Jastrow factor implementation."""

import jax
import jax.numpy as jnp

from jaqmc.wavefunction.jastrow import SimpleEEJastrow, _cusp_function


def _compute_r_ee(electrons):
    diff = electrons[:, None, :] - electrons[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


class TestCuspFunction:
    """Tests for the Jastrow cusp function."""

    def test_cusp_function_derivative_at_zero(self):
        """Test that the cusp function satisfies the cusp condition at r=0.

        The cusp condition requires:
            d f(r) / dr |_{r=0} = c

        For f(r) = -(c * alpha^2) / (alpha + r), we have:
            d f(r) / dr = (c * alpha^2) / (alpha + r)^2
            d f(r) / dr |_{r=0} = (c * alpha^2) / alpha^2 = c

        This test verifies the derivative analytically using JAX autodiff.
        """
        alpha = jnp.array([1.5])

        for c in [0.25, 0.5]:
            # Use JAX grad for analytical derivative
            def f(r):
                return _cusp_function(r, c, alpha)[0]

            df_dr = jax.grad(f)(jnp.array(0.0))

            # Should be exactly equal to c value
            assert jnp.allclose(df_dr, c, atol=1e-5)


class TestSimpleEEJastrow:
    """Tests for SimpleEEJastrow module."""

    def test_jastrow_cusp_derivative_parallel_spins(self):
        """Test Jastrow provides correct cusp for parallel spins (dJ/dr = 0.25)."""
        nspins = (2, 0)  # Only parallel pairs
        jastrow = SimpleEEJastrow(nspins=nspins)

        key = jax.random.PRNGKey(0)
        # Initialize params at a moderate distance
        r_init = 1.0
        r_ee_init = jnp.array([[0.0, r_init], [r_init, 0.0]])
        params = jastrow.init(key, r_ee_init)

        # Test at small inter-electron distance
        r_small = 0.001

        def jastrow_of_r(r):
            # Directly construct r_ee matrix for 2 electrons at distance r
            # This avoids sqrt(0) gradient issues from self-distance computation
            r_ee = jnp.array([[0.0, r], [r, 0.0]])
            return jastrow.apply(params, r_ee)

        dJ_dr = jax.grad(jastrow_of_r)(r_small)
        assert jnp.allclose(dJ_dr, 0.25, atol=1e-3)

    def test_jastrow_cusp_derivative_antiparallel_spins(self):
        """Test Jastrow provides correct cusp for antiparallel spins (dJ/dr = 0.5)."""
        nspins = (1, 1)  # One up, one down (antiparallel)
        jastrow = SimpleEEJastrow(nspins=nspins)

        key = jax.random.PRNGKey(0)
        # Initialize params at a moderate distance
        r_init = 1.0
        r_ee_init = jnp.array([[0.0, r_init], [r_init, 0.0]])
        params = jastrow.init(key, r_ee_init)

        # Test at small inter-electron distance
        r_small = 0.001

        def jastrow_of_r(r):
            # Directly construct r_ee matrix for 2 electrons at distance r
            r_ee = jnp.array([[0.0, r], [r, 0.0]])
            return jastrow.apply(params, r_ee)

        dJ_dr = jax.grad(jastrow_of_r)(r_small)
        assert jnp.allclose(dJ_dr, 0.5, atol=1e-3)
