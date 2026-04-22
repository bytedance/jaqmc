# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Kinetic energy estimator in Euclidean geometry."""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import LocalEstimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import (
    grad_maybe_complex,
    linearize_maybe_complex,
    transform_with_data,
)
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

from ._common import LaplacianMode, _apply_kinetic_formula, _flatten_positions


@configurable_dataclass
class EuclideanKinetic(LocalEstimator):
    """Kinetic energy estimator in Euclidean geometry.

    The most computationally expensive default energy component. The
    ``mode`` setting controls how the diagonal Hessian is computed and
    is the main performance knob — see :class:`LaplacianMode` for
    trade-offs.

    .. seealso:: :doc:`/guide/estimators/kinetic` for the derivation
       and Laplacian computation details.

    Args:
        mode: Laplacian computation strategy. ``forward_laplacian`` is the default
            for JAX 0.7.1 and later, ``scan`` for earlier versions. See
            :class:`LaplacianMode` for details.
        sparsity_threshold: Sparsity threshold when using ``forward_laplacian`` mode.
            Always verify numerical correctness before adopting it in production runs.
        f_log_psi: Log-psi evaluate function (runtime dep).
        data_field: Name of the data field containing positions (runtime dep).
    """

    mode: LaplacianMode = (
        LaplacianMode.scan
        if jax.__version_info__ < (0, 7, 1)
        else LaplacianMode.forward_laplacian
    )
    sparsity_threshold: int = 0
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def __post_init__(self):
        if self.mode == LaplacianMode.forward_laplacian:
            if jax.__version_info__ < (0, 7, 1):
                raise RuntimeError(
                    "JAX version too old to run folx. "
                    "Please upgrade to JAX 0.7.1 or later."
                )
        elif self.sparsity_threshold > 0:
            raise ValueError(
                "Sparsity threshold is only supported in forward_laplacian mode."
            )

    def evaluate_local(
        self,
        params: Params,
        data: Data,
        prev_local_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_local_stats, rngs
        if self.mode == LaplacianMode.forward_laplacian:
            return self._evaluate_forward_laplacian(params, data, state)
        return self._evaluate_standard(params, data, state)

    def _evaluate_standard(
        self, params: Params, data: Data, state: None
    ) -> tuple[dict[str, Any], None]:
        grad_f = transform_with_data(
            self.f_log_psi, self.data_field, grad_maybe_complex
        )
        flatten_positions, positions_shape = _flatten_positions(data, self.data_field)
        n = flatten_positions.size

        def grad_f_closure(x):
            return grad_f(
                params, data.merge({self.data_field: jnp.reshape(x, positions_shape)})
            ).flatten()

        primal, dgrad_f = linearize_maybe_complex(grad_f_closure, flatten_positions)

        eye = parallel_jax.pvary(jnp.eye(n))
        if self.mode == LaplacianMode.scan:
            _, diagonal = jax.lax.scan(
                lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n
            )
            laplacian = jnp.sum(diagonal)
        else:
            laplacian = jax.lax.fori_loop(
                0, n, lambda i, val: val + dgrad_f(eye[i])[i], 0.0
            )

        result = _apply_kinetic_formula(laplacian, jnp.sum(primal**2))
        return {"energy:kinetic": result}, state

    def _evaluate_forward_laplacian(
        self, params: Params, data: Data, state: None
    ) -> tuple[dict[str, Any], None]:
        from folx import forward_laplacian

        if jax.__version_info__ < (0, 7, 1):
            raise RuntimeError("JAX version too old to run folx.")

        flatten_positions, positions_shape = _flatten_positions(data, self.data_field)

        def log_psi_closure(x):
            return self.f_log_psi(
                params, data.merge({self.data_field: jnp.reshape(x, positions_shape)})
            )

        log_psi_folx = forward_laplacian(
            log_psi_closure, sparsity_threshold=self.sparsity_threshold
        )
        fwd_result = log_psi_folx(flatten_positions)
        laplacian = fwd_result.laplacian
        primal = fwd_result.dense_jacobian

        result = _apply_kinetic_formula(laplacian, jnp.sum(primal**2))
        return {"energy:kinetic": result}, state
