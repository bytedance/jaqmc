# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Kinetic energy estimator in Euclidean geometry."""

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
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
class EuclideanKinetic(PerWalkerEstimator):
    r"""Kinetic energy estimator in Euclidean geometry.

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
        prefactor: Scalar or per-particle factor multiplying the kinetic energy.
            Defaults to ``1.0`` for the standard :math:`-\tfrac{1}{2}\nabla^2`
            operator. Pass a scalar ``1 / m`` for a uniform effective mass, or
            fold in a unit conversion (e.g. the moire model uses
            ``hartree_to_mev / (mass * a^2)`` to map the dimensionless Laplacian
            to meV). Pass a shape ``(n_particles,)`` array to weight each
            particle separately (e.g. per-particle effective masses); the
            per-particle factor is applied via a coordinate rescaling and works
            in every :class:`LaplacianMode`.
        sparse: Whether to seed the Forward Laplacian path with a sparse
            particle-coordinate input so the interpreter can preserve locality
            internally where possible before returning a dense public Jacobian.
        f_log_psi: Log-psi evaluate function (runtime dep).
        data_field: Name of the data field containing positions (runtime dep).
    """

    mode: LaplacianMode = (
        LaplacianMode.scan
        if jax.__version_info__ < (0, 7, 1)
        else LaplacianMode.forward_laplacian
    )
    prefactor: float | list[float] = 1.0
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")
    sparse: bool = True

    def __post_init__(self):
        if self.mode == LaplacianMode.forward_laplacian and jax.__version_info__ < (
            0,
            7,
            1,
        ):
            raise RuntimeError(
                "JAX version too old to run jaqmc.laplacian. "
                "Please upgrade to JAX 0.7.1 or later."
            )

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        prefactor = jnp.asarray(self.prefactor)
        if prefactor.ndim != 0:
            # Per-particle prefactor: fold sqrt(prefactor) into a coordinate
            # rescaling and evaluate with unit prefactor. With s = sqrt(w), the
            # unit-prefactor kinetic energy of log_psi(s * y) at y = x / s equals
            # -0.5 * sum_p w_p [lap_p + |grad_p|^2], since each s_p feeds
            # s_p^2 = w_p into both the diagonal Hessian and the squared gradient
            # of particle p. This reuses the scalar path for every LaplacianMode.
            positions = data[self.data_field]
            scale = jnp.sqrt(jnp.asarray(self.prefactor, dtype=positions.dtype))
            scale = scale.reshape(scale.shape + (1,) * (positions.ndim - scale.ndim))
            f_log_psi = self.f_log_psi
            data_field = self.data_field

            def scaled_log_psi(params: Params, inner_data: Data) -> jnp.ndarray:
                return f_log_psi(
                    params,
                    inner_data.merge({data_field: scale * inner_data[data_field]}),
                )

            rescaled = dataclasses.replace(
                self,
                prefactor=1.0,
                f_log_psi=cast(NumericWavefunctionEvaluate, scaled_log_psi),
            )
            return rescaled.evaluate_single_walker(
                params,
                data.merge({data_field: positions / scale}),
                prev_walker_stats,
                state,
                rngs,
            )
        del prev_walker_stats, rngs
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
        return {"energy:kinetic": self.prefactor * result}, state

    def _evaluate_forward_laplacian(
        self, params: Params, data: Data, state: None
    ) -> tuple[dict[str, Any], None]:
        from jaqmc.laplacian import forward_laplacian, make_laplacian_input

        if jax.__version_info__ < (0, 7, 1):
            raise RuntimeError("JAX version too old to run jaqmc.laplacian.")

        positions = make_laplacian_input(
            data[self.data_field],
            sparse_axis=0 if self.sparse else None,
        )
        fwd_result = forward_laplacian(self.f_log_psi)(
            params,
            data.merge({self.data_field: positions}),
        )
        laplacian = fwd_result.laplacian
        primal = fwd_result.dense_jacobian
        grad_sq = jnp.sum(primal**2)

        result = _apply_kinetic_formula(laplacian, grad_sq)
        return {"energy:kinetic": self.prefactor * result}, state
