# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from jaqmc.utils.subspace_linalg import (
    complex_variance,
    generalized_ritz,
    rayleigh_solve,
    row_scaled_matrix,
    solve_residual,
    stable_complex_logdet,
)


def test_row_scaling_and_complex_logdet_are_stable():
    logs = jnp.array([[1000.0, 999.0], [998.0, 1001.0]], dtype=jnp.complex64)
    scaled, shifts = row_scaled_matrix(logs)
    expected = jnp.linalg.slogdet(scaled)

    actual = stable_complex_logdet(logs)

    np.testing.assert_allclose(actual.real, shifts.sum() + expected[1], rtol=1e-6)
    np.testing.assert_allclose(actual.imag, jnp.angle(expected[0]), atol=1e-6)
    assert np.isfinite(actual)


def test_rayleigh_solve_and_residual():
    phi = jnp.array([[2.0, 0.5], [0.25, 1.5]], dtype=jnp.complex64)
    expected = jnp.array([[1.0, 0.2j], [-0.1j, 3.0]], dtype=jnp.complex64)
    phi_h = phi @ expected

    actual = rayleigh_solve(phi, phi_h)

    # CUDA complex64 small-matrix solves can use lower-precision kernels than
    # the CPU backend.  Keep this test about dtype-appropriate correctness;
    # production Rayleigh evaluation defaults to complex128.
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(solve_residual(phi, actual, phi_h), 0, atol=5e-4)


def test_generalized_ritz_matches_explicit_reference():
    g = jnp.array([[2.0, 0.2], [0.2, 1.0]])
    gh = jnp.array([[1.0, 0.1], [0.1, 3.0]])

    values, vectors = generalized_ritz(g, gh)
    residual = gh @ vectors - g @ vectors * values[None, :]

    np.testing.assert_allclose(residual, 0, atol=2e-6)
    np.testing.assert_allclose(vectors.T @ g @ vectors, jnp.eye(2), atol=2e-6)


def test_complex_variance_uses_absolute_square():
    samples = jnp.array([1 + 1j, 1 - 1j])
    np.testing.assert_allclose(complex_variance(samples), 1.0)


def test_rayleigh_similarity_preserves_trace_and_spectrum():
    rayleigh = jnp.array([[1.0, 0.2], [0.3, 2.0]], dtype=jnp.complex64)
    basis = jnp.array([[1.0, 0.4], [-0.2, 1.0]], dtype=jnp.complex64)
    transformed = jnp.linalg.solve(basis, rayleigh @ basis)

    np.testing.assert_allclose(
        jnp.trace(transformed), jnp.trace(rayleigh), atol=5e-4
    )
    np.testing.assert_allclose(
        jnp.sort(jnp.linalg.eigvals(transformed)),
        jnp.sort(jnp.linalg.eigvals(rayleigh)),
        atol=5e-4,
    )
