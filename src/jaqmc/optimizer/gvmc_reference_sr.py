# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Small-system reference SR backend for determinant-state Grassmann VMC.

The production Grassmann optimizer is JaQMC's existing distributed
``SROptimizer`` applied to the determinant-state log amplitude.  This module
implements the sample-space equations used by the accompanying GVMC research
code as a transparent numerical oracle and A/B backend.  It deliberately does
not replace or modify JaQMC's SR implementation.  The reference snapshot is
``cqsl/GVMC@e68e503``; the equations are independently expressed here because
that repository does not currently declare a software license.
"""

import math
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.flatten_util import ravel_pytree

from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.utils.chunked_vmap import chunked_vmap
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import grad_maybe_complex
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

__all__ = [
    "GVMCReferenceSROptimizer",
    "GVMCReferenceSRState",
    "minsr_solve",
    "minsr_solve_gradient",
    "minsr_solve_kacz",
]


def _native_vmc_to_gvmc_gradient(gradient: jax.Array) -> jax.Array:
    r"""Convert JaQMC's energy-gradient convention to the GVMC force form.

    JaQMC's VMC estimators return ``2 Re[J^H B]`` for real parameters, while
    the GVMC minimum-SR source equation is written for ``Re[J^H B]``.  Keep
    this convention conversion at the reference-backend adapter boundary so
    neither the native estimator nor the generic optimizer contract changes.

    Returns:
        The gradient in the GVMC reference convention.
    """
    return 0.5 * gradient


def _sample_matrix(jacobian: jax.Array, lam0: float, lam1: float) -> jax.Array:
    """Build GVMC's matrix; ``lam1`` shifts all entries, not the diagonal.

    Returns:
        The regularized sample-space matrix.
    """
    n_samples = jacobian.shape[0]
    scale = jnp.linalg.norm(jacobian) ** 2 / n_samples
    matrix = jacobian @ jnp.conj(jacobian.T)
    matrix = matrix + scale * (lam1 / n_samples)
    return matrix + lam0 * jnp.eye(n_samples, dtype=matrix.dtype)


def minsr_solve(
    jacobian: jax.Array,
    force: jax.Array,
    *,
    lam0: float = 1e-4,
    lam1: float = 1.0,
) -> jax.Array:
    r"""Solve the GVMC sample-space minimum-SR equation.

    Computes ``J^H A^-1 force`` with
    ``A = J J^H + lam1 * ||J||^2 / n^2 + lam0 I``.  The ``lam1`` term is the
    scalar all-ones shift used by the reference implementation, rather than a
    second diagonal shift.

    Returns:
        The minimum-SR parameter direction.

    Raises:
        ValueError: If the Jacobian or force has an incompatible shape.
    """
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a rank-2 matrix")
    if force.shape != (jacobian.shape[0],):
        raise ValueError(
            f"force must have shape {(jacobian.shape[0],)}, got {force.shape}"
        )
    factor = jsp.linalg.cho_factor(_sample_matrix(jacobian, lam0, lam1))
    return jnp.conj(jacobian.T) @ jsp.linalg.cho_solve(factor, force)


def minsr_solve_kacz(
    jacobian: jax.Array,
    force: jax.Array,
    previous_delta: jax.Array,
    *,
    lam0: float = 1e-4,
    lam1: float = 1.0,
    mu: float = 0.99,
) -> jax.Array:
    """Apply the GVMC Kaczmarz/SPRING continuation equation.

    Returns:
        The continued minimum-SR parameter direction.
    """
    residual_force = force - mu * (jacobian @ previous_delta)
    return (
        minsr_solve(jacobian, residual_force, lam0=lam0, lam1=lam1)
        + mu * previous_delta
    )


def minsr_solve_gradient(
    jacobian: jax.Array,
    gradient: jax.Array,
    *,
    lam0: float = 1e-4,
    lam1: float = 1.0,
) -> jax.Array:
    r"""Apply the same minimum-SR solve to an already reduced VMC gradient.

    If ``gradient = J^H force``, the matrix inversion lemma makes this exactly
    equivalent to :func:`minsr_solve`.  This form preserves JaQMC's native
    ``OptimizerLike`` contract, whose optimizer receives a parameter-shaped
    energy gradient rather than the per-walker local-energy force.

    Returns:
        The minimum-SR direction reconstructed from the reduced gradient.

    Raises:
        ValueError: If regularization is invalid or gradient shape is incompatible.
    """
    if lam0 <= 0:
        raise ValueError("lam0 must be positive for the gradient-form solve")
    if gradient.shape != (jacobian.shape[1],):
        raise ValueError(
            f"gradient must have shape {(jacobian.shape[1],)}, got {gradient.shape}"
        )
    factor = jsp.linalg.cho_factor(_sample_matrix(jacobian, lam0, lam1))
    projected = jacobian @ gradient
    correction = jnp.conj(jacobian.T) @ jsp.linalg.cho_solve(factor, projected)
    return (gradient - correction) / lam0


class GVMCReferenceSRState(NamedTuple):
    """Optimizer state containing the step counter and previous SR direction."""

    counter: jax.Array
    previous_delta: jax.Array


@configurable_dataclass
class GVMCReferenceSROptimizer:
    """Source-equation Grassmann SR backend for single-device A/B tests.

    For production, use :class:`jaqmc.optimizer.sr.SROptimizer`; it evaluates
    the same Grassmann score while adding distributed reductions, chunked Gram
    construction, mixed precision, and robust stabilization.  The native
    ``OptimizerLike`` gradient supplied to :meth:`update` follows JaQMC's
    ``2 Re[J^H B]`` VMC convention.  This adapter divides it by two before
    applying the GVMC source equation, whose reduced gradient is
    ``Re[J^H B]``.  ``previous_delta`` is therefore stored in GVMC convention.

    Args:
        learning_rate: Constant update step used by the GVMC reference code.
        lam0: Diagonal sample-space regularization.
        lam1: Centered-score null-direction regularization.
        mu: Kaczmarz/SPRING continuation coefficient.
        score_chunk_size: Optional chunk size for score evaluation.
        f_log_psi: Runtime-wired determinant log amplitude.
    """

    learning_rate: float = 0.1
    lam0: float = 1e-3
    lam1: float = 1.0
    mu: float = 0.9
    score_chunk_size: int | None = None
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lam0 <= 0:
            raise ValueError("lam0 must be positive")
        if self.lam1 < 0:
            raise ValueError("lam1 must be non-negative")
        if not 0 <= self.mu < 1:
            raise ValueError("mu must satisfy 0 <= mu < 1")
        if self.score_chunk_size is not None and self.score_chunk_size < 1:
            raise ValueError("score_chunk_size must be positive or None")

    def init(
        self,
        params: Params,
        *,
        batched_data: BatchedData,
        **extra: Any,
    ) -> GVMCReferenceSRState:
        """Initialize the previous-direction state.

        Returns:
            A zero-initialized reference optimizer state.

        Raises:
            NotImplementedError: If multiple devices or complex parameters are used.
        """
        del batched_data, extra
        if jax.device_count() != 1:
            raise NotImplementedError(
                "GVMCReferenceSROptimizer is a single-device numerical oracle; "
                "use jaqmc.optimizer.sr:SROptimizer for multi-device runs."
            )
        flat, _ = ravel_pytree(params)
        if jnp.iscomplexobj(flat):
            raise NotImplementedError(
                "GVMCReferenceSROptimizer currently requires real parameters."
            )
        return GVMCReferenceSRState(
            counter=jnp.asarray(0, dtype=jnp.int32),
            previous_delta=jnp.zeros_like(flat),
        )

    def _score_matrix(self, params: Params, batched_data: BatchedData) -> jax.Array:
        score_fn = grad_maybe_complex(self.f_log_psi)
        score_tree = chunked_vmap(
            lambda sample: score_fn(params, sample),
            in_axes=(batched_data.vmap_axis,),
            out_axes=0,
            chunk_size=self.score_chunk_size,
        )(batched_data.data)
        score = jax.vmap(lambda tree: ravel_pytree(tree)[0])(score_tree)
        score = (score - jnp.mean(score, axis=0, keepdims=True)) / math.sqrt(
            batched_data.batch_size
        )
        if jnp.iscomplexobj(score):
            score = jnp.concatenate((jnp.real(score), jnp.imag(score)), axis=0)
        return score

    def update(
        self,
        grads: Params,
        state: GVMCReferenceSRState,
        params: Params,
        *,
        batched_data: BatchedData,
        **extra: Any,
    ) -> tuple[Params, GVMCReferenceSRState]:
        """Return a reference Grassmann-SR parameter update."""
        del extra
        native_gradient, unravel = ravel_pytree(grads)
        gradient = _native_vmc_to_gvmc_gradient(native_gradient)
        score = self._score_matrix(params, batched_data)
        metric_previous = score @ state.previous_delta
        residual_gradient = gradient - self.mu * (jnp.conj(score.T) @ metric_previous)
        delta = (
            minsr_solve_gradient(
                score,
                residual_gradient,
                lam0=self.lam0,
                lam1=self.lam1,
            )
            + self.mu * state.previous_delta
        )
        # Match cqsl/GVMC's actual parameter update.  Its lr / sqrt(M)
        # quantity is used only for an auxiliary displacement diagnostic and
        # is not applied to params.
        updates = unravel(-self.learning_rate * delta)
        return updates, GVMCReferenceSRState(
            counter=state.counter + 1,
            previous_delta=delta,
        )
