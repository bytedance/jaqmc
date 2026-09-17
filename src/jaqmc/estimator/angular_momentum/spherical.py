# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Total angular-momentum estimator on a Haldane sphere.

Computes local :math:`L_z`, :math:`L_z^2`, and :math:`L^2` observables for a
wavefunction in the magnetic field of a central monopole.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.geometry.sphere import spinor_coordinates_from_angles
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


def _rotate_spinors_about_axis(
    u: jnp.ndarray, v: jnp.ndarray, angle: jnp.ndarray, axis: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the same single-axis rotation to every electron spinor.

    Returns:
        The rotated ``u`` and ``v`` spinor components.
    """
    spinors = jnp.stack([u, v], axis=-1)
    sin_half = jnp.sin(angle / 2)
    cos_half = jnp.cos(angle / 2)
    if axis == 0:
        rotation = jnp.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]])
    if axis == 1:
        rotation = jnp.array([[cos_half, -sin_half], [sin_half, cos_half]])
    if axis == 2:
        zero = jnp.zeros_like(cos_half)
        rotation = jnp.array(
            [[cos_half - 1j * sin_half, zero], [zero, cos_half + 1j * sin_half]]
        )
    # Row-vector convention: row @ U.T applies U to each spinor column.
    rotated = spinors @ rotation.T
    return rotated[..., 0], rotated[..., 1]


def _first_and_second_derivative(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    direction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate two derivative orders of ``fn`` along one direction.

    Returns:
        The first and second directional derivatives.
    """

    def first_derivative(y: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(fn, (y,), (direction,))[1]

    first, second = jax.jvp(first_derivative, (x,), (direction,))
    return first, second


@configurable_dataclass
class SphericalAngularMomentum(PerWalkerEstimator):
    r"""Local total-angular-momentum estimator on a Haldane sphere.

    The estimator differentiates one scalar SU(2) dummy rotation per Cartesian
    axis, applied globally to all monopole spinors. If ``g_a(t) = log psi(U_a(t)
    z_i)`` for every electron, then ``L_a = i d/dt`` at ``t = 0`` and

    .. math::

        \frac{L^2\psi}{\psi} =
            -\sum_a (g_a')^2 - \sum_a g_a''.

    This evaluates the conserved total angular momentum directly in spinor
    coordinates, avoiding coordinate singularities at the poles.

    Args:
        f_log_psi_from_spinor: Complex log-psi ``(params, u, v)`` (runtime dep).
        data_field: Electron-coordinate field name (runtime dep).
    """

    f_log_psi_from_spinor: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        runtime_dep()
    )
    data_field: str = runtime_dep(default="electrons")

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        electrons = data[self.data_field]
        u, v = spinor_coordinates_from_angles(electrons)
        angle_dtype = jnp.result_type(jnp.real(u).dtype, jnp.float32)
        zero_angle = jnp.zeros((), dtype=angle_dtype)
        unit_direction = jnp.ones((), dtype=angle_dtype)

        def rotated_log_psi(axis: int, angle: jnp.ndarray) -> jnp.ndarray:
            rotated_u, rotated_v = _rotate_spinors_about_axis(u, v, angle, axis)
            return self.f_log_psi_from_spinor(params, rotated_u, rotated_v)

        # One scalar dummy angle per Cartesian axis; only diagonal derivatives
        # at zero enter Lz, Lz^2, and L^2.
        pairs = [
            _first_and_second_derivative(
                partial(rotated_log_psi, axis), zero_angle, unit_direction
            )
            for axis in range(3)
        ]
        gradient = jnp.stack([first for first, _ in pairs])
        hessian_diagonal = jnp.stack([second for _, second in pairs])
        laplacian = jnp.sum(hessian_diagonal)
        second_z = hessian_diagonal[2]
        angular_momentum_z = 1j * gradient[2]
        angular_momentum_z_square = -(gradient[2] ** 2 + second_z)
        angular_momentum_square = -(jnp.sum(gradient**2) + laplacian)
        return {
            "angular_momentum_z": angular_momentum_z.real,
            "angular_momentum_z_square": angular_momentum_z_square.real,
            "angular_momentum_square": angular_momentum_square.real,
        }, state
