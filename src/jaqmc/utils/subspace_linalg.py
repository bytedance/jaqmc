# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Numerically stable small-matrix operations for variational subspaces."""

import jax
from jax import numpy as jnp


def row_scaled_matrix(log_amplitudes: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Exponentiate a log-amplitude matrix after removing per-row scales."""
    row_shift = jax.lax.stop_gradient(
        jnp.max(jnp.real(log_amplitudes), axis=-1)
    )
    return jnp.exp(log_amplitudes - row_shift[:, None]), row_shift


def stable_complex_logdet(log_amplitudes: jax.Array) -> jax.Array:
    """Return ``log(det(exp(log_amplitudes)))`` without amplitude overflow."""
    scaled, row_shift = row_scaled_matrix(log_amplitudes)
    phase, logabsdet = jnp.linalg.slogdet(scaled)
    return jnp.sum(row_shift) + logabsdet + 1j * jnp.angle(phase)


def rayleigh_solve(phi: jax.Array, phi_h: jax.Array) -> jax.Array:
    """Solve the local Rayleigh system ``phi @ R = phi_h``."""
    return jnp.linalg.solve(phi, phi_h)


def solve_residual(a: jax.Array, x: jax.Array, b: jax.Array) -> jax.Array:
    """Return the relative Frobenius residual of ``a @ x = b``."""
    tiny = jnp.finfo(jnp.real(b).dtype).tiny
    return jnp.linalg.norm(a @ x - b) / jnp.maximum(jnp.linalg.norm(b), tiny)


def complex_variance(x: jax.Array, axis=0) -> jax.Array:
    """Return ``E[|x-E[x]|^2]`` for real or complex samples."""
    centered = x - jnp.nanmean(x, axis=axis, keepdims=True)
    return jnp.nanmean(jnp.abs(centered) ** 2, axis=axis)


def matrix_condition_proxy(matrix: jax.Array) -> jax.Array:
    """Return the singular-value condition number of a small matrix."""
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    return singular_values[0] / singular_values[-1]


def singular_value_diagnostics(
    matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return minimum/maximum singular values and their condition ratio."""
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]
    return sigma_min, sigma_max, sigma_max / sigma_min


def hermitianized_rayleigh(g: jax.Array, gh: jax.Array) -> jax.Array:
    """Map a generalized Hermitian eigenproblem to an ordinary one."""
    g = (g + g.conj().T) / 2
    gh = (gh + gh.conj().T) / 2
    chol = jnp.linalg.cholesky(g)
    left_solved = jnp.linalg.solve(chol, gh)
    result = jnp.linalg.solve(chol, left_solved.conj().T).conj().T
    return (result + result.conj().T) / 2


def generalized_ritz(g: jax.Array, gh: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve ``GH c = G c e`` for positive-definite overlap ``G``."""
    effective = hermitianized_rayleigh(g, gh)
    values, orthonormal_vectors = jnp.linalg.eigh(effective)
    chol = jnp.linalg.cholesky((g + g.conj().T) / 2)
    vectors = jnp.linalg.solve(chol.conj().T, orthonormal_vectors)
    return values, vectors
