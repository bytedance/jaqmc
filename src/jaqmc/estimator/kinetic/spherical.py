# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Kinetic energy estimator on a sphere with magnetic monopole.

Computes :math:`\Lambda^2 / (2R^2)` on a Haldane sphere with monopole
strength :math:`Q`.

.. seealso:: :doc:`/guide/estimators/kinetic` for the full formulation.
"""

from collections.abc import Callable, Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.geometry.sphere import (
    spinor_coordinates_from_stereographic,
    stereographic_coordinates,
    stereographic_monopole_connection,
)
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import grad_maybe_complex, linearize_maybe_complex
from jaqmc.utils.wiring import runtime_dep

from ._common import (
    LaplacianMode,
    default_laplacian_mode,
    hessian_diagonal_laplacian,
    require_forward_laplacian,
)


@configurable_dataclass
class SphericalKinetic(PerWalkerEstimator):
    r"""Local kinetic-energy estimator on a sphere.

    Kinetic energy uses nearest-pole stereographic derivatives and a local
    monopole gauge, making the calculation regular at the poles.

    Args:
        monopole_strength: Monopole strength :math:`Q = \mathrm{flux}/2`.
        radius: Sphere radius. Defaults to :math:`\sqrt{Q}` for ``Q > 0``.
        mode: Laplacian computation strategy. ``forward_laplacian`` is the default
            for JAX 0.7.1 and later, ``scan`` for earlier versions. See
            :class:`LaplacianMode` for details.
        f_log_psi_from_spinor: Complex log-psi ``(params, u, v)`` (runtime dep).
        data_field: Electron-coordinate field name.
    """

    monopole_strength: float = 1.0
    radius: float | None = None
    mode: LaplacianMode = default_laplacian_mode()
    f_log_psi_from_spinor: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        runtime_dep()
    )
    data_field: str = runtime_dep(default="electrons")

    def __post_init__(self):
        require_forward_laplacian(self.mode)

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs

        Q = self.monopole_strength
        electrons = data[self.data_field]
        plane, is_north = stereographic_coordinates(electrons)
        sqrt_metric = (1 + jnp.sum(plane**2, axis=-1)) / 2
        metric = sqrt_metric**2
        radius = jnp.array(self.radius if self.radius is not None else jnp.sqrt(Q))
        connection = stereographic_monopole_connection(plane, is_north, Q)

        def f(p: Params, x: jnp.ndarray) -> jnp.ndarray:
            u, v = spinor_coordinates_from_stereographic(x, is_north)
            return self.f_log_psi_from_spinor(p, u, v)

        laplacian, gradient = (
            self._evaluate_forward_laplacian(params, f, plane, sqrt_metric)
            if self.mode == LaplacianMode.forward_laplacian
            else self._evaluate_standard_laplacian(params, f, plane, metric)
        )

        covariant_laplacian_logpsi = (
            laplacian
            + jnp.sum(metric * jnp.sum(gradient**2, axis=-1))
            - 2j * jnp.sum(metric * jnp.sum(connection * gradient, axis=-1))
            - jnp.sum(metric * jnp.sum(connection**2, axis=-1))
        )
        kinetic_energy = -covariant_laplacian_logpsi / (2 * radius**2)

        return {"energy:kinetic": kinetic_energy}, state

    def _evaluate_forward_laplacian(
        self,
        params: Params,
        f: Callable[[Params, jnp.ndarray], jnp.ndarray],
        plane: jnp.ndarray,
        sqrt_metric: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        from jaqmc.laplacian import forward_laplacian, make_laplacian_input

        # Seed Forward Laplacian with sqrt(g) so its laplacian output already
        # carries the metric stretch; undo that scale on the returned gradient.
        metric_weight = sqrt_metric[:, None]
        fwdlap_output = forward_laplacian(f)(
            params,
            make_laplacian_input(plane, weights=metric_weight, sparse_axis=0),
        )
        gradient = fwdlap_output.dense_jacobian.reshape(plane.shape) / metric_weight
        return fwdlap_output.laplacian, gradient

    def _evaluate_standard_laplacian(
        self,
        params: Params,
        f: Callable[[Params, jnp.ndarray], jnp.ndarray],
        plane: jnp.ndarray,
        metric: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gradient_fn = grad_maybe_complex(f, argnums=1)
        flatten_plane = plane.flatten()

        def gradient_closure(x: jnp.ndarray) -> jnp.ndarray:
            return gradient_fn(params, jnp.reshape(x, plane.shape)).flatten()

        gradient, gradient_jvp = linearize_maybe_complex(
            gradient_closure, flatten_plane
        )
        laplacian = hessian_diagonal_laplacian(
            gradient_jvp,
            flatten_plane.size,
            self.mode,
            weights=jnp.broadcast_to(metric[..., None], plane.shape).flatten(),
        )
        return laplacian, gradient.reshape(plane.shape)
