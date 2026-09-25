# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Angular-momentum estimators on a Haldane sphere.

Provides local :math:`L_z`, :math:`L_z^2`, and :math:`L^2` observables, and a
projection onto total-angular-momentum sectors.
"""

import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.geometry.sphere import sample_haar_su2, spinor_coordinates_from_angles
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


def _angular_momentum_characters(
    half_traces: jnp.ndarray, twice_angular_momenta: tuple[int, ...]
) -> jnp.ndarray:
    r"""Return :math:`\chi_L(g)=U_{2L}(x)` for each requested sector.

    :math:`x` is half the trace of the fundamental representation of
    :math:`g\in SU(2)`. The Chebyshev polynomials of the second kind are the
    characters of the angular-momentum irreps.

    Args:
        half_traces: Half-traces :math:`x` with shape ``(n_rotations,)``.
        twice_angular_momenta: Values of :math:`2L` for the requested sectors.

    Returns:
        Characters with shape ``(n_sectors, n_rotations)``.
    """
    max_degree = max(twice_angular_momenta)
    polynomials = [jnp.ones_like(half_traces)]
    if max_degree:
        polynomials.append(2 * half_traces)
    for _ in range(2, max_degree + 1):
        polynomials.append(2 * half_traces * polynomials[-1] - polynomials[-2])
    return jnp.stack([polynomials[degree] for degree in twice_angular_momenta])


@configurable_dataclass
class SphericalAngularMomentumProjection(PerWalkerEstimator):
    r"""Project a Hall wavefunction onto total-angular-momentum sectors.

    For Haar-distributed rotations :math:`g\in SU(2)`, the estimator averages
    character-weighted rotational overlaps

    .. math::

        p_L &= (2L+1)\left\langle
            \chi_L(g)\frac{\psi(g^{-1}R)}{\psi(R)}
        \right\rangle_{|\psi|^2,g},\\
        h_L &= (2L+1)\left\langle
            \chi_L(g)\, E_{\mathrm{loc}}(R)^*
            \frac{\psi(g^{-1}R)}{\psi(R)}
        \right\rangle_{|\psi|^2,g}.

    :math:`U(g)` acts by rotating every electron's monopole spinor. Each walker
    evaluation draws a fresh batch of Haar rotations, so increasing the walker
    count or the number of evaluation steps improves the Haar integral instead
    of refining a fixed rotation quadrature. :meth:`finalize_stats` reports
    :math:`p_L` and :math:`E_L=h_L/p_L`.

    Requires ``total_energy`` in ``prev_walker_stats`` from a preceding energy
    estimator.

    Args:
        angular_momenta: Nonnegative integer or half-integer total angular
            momenta to resolve.
        n_rotations: Number of fresh Haar rotations sampled per walker
            evaluation.
        f_log_psi_from_spinor: Complex log-psi ``(params, u, v)`` (runtime dep).
        data_field: Electron-coordinate field name (runtime dep).
    """

    angular_momenta: tuple[float, ...] = (0.0,)
    n_rotations: int = 256
    f_log_psi_from_spinor: Callable[[Params, jnp.ndarray, jnp.ndarray], jnp.ndarray] = (
        runtime_dep()
    )
    data_field: str = runtime_dep(default="electrons")

    def __post_init__(self) -> None:
        if not self.angular_momenta:
            raise ValueError("angular_momenta must contain at least one sector.")
        if self.n_rotations <= 0:
            raise ValueError("n_rotations must be positive.")

        doubled = tuple(
            2.0 * angular_momentum for angular_momentum in self.angular_momenta
        )
        if any(
            not math.isfinite(value) or value < 0 or not value.is_integer()
            for value in doubled
        ):
            raise ValueError(
                "angular_momenta must contain finite, nonnegative integer "
                "or half-integer values."
            )
        twice_angular_momenta = tuple(int(value) for value in doubled)
        if len(set(twice_angular_momenta)) != len(twice_angular_momenta):
            raise ValueError("angular_momenta must not contain duplicates.")
        self._twice_angular_momenta = twice_angular_momenta

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, jnp.ndarray], None]:
        r"""Average character-weighted overlaps over ``n_rotations`` Haar samples.

        Returns:
            ``angular_projection:weight`` (:math:`p_L`) and
            ``angular_projection:energy_weight`` (:math:`h_L`) for this walker,
            plus unchanged state.
        """
        electrons = data[self.data_field]
        u, v = spinor_coordinates_from_angles(electrons)
        log_psi = self.f_log_psi_from_spinor(params, u, v)
        spinors = jnp.stack([u, v], axis=-1)
        rotations = sample_haar_su2(rngs, self.n_rotations, electrons.dtype)
        inverse_rotations = jnp.swapaxes(jnp.conj(rotations), -1, -2)
        fundamental_half_traces = jnp.real(jnp.trace(rotations, axis1=-2, axis2=-1)) / 2
        characters = _angular_momentum_characters(
            fundamental_half_traces, self._twice_angular_momenta
        )
        rotated_spinors = jnp.einsum("rab,ib->ria", inverse_rotations, spinors)

        def rotated_log_psi(rotated: jnp.ndarray) -> jnp.ndarray:
            rotated_u, rotated_v = rotated[..., 0], rotated[..., 1]
            return self.f_log_psi_from_spinor(params, rotated_u, rotated_v)

        log_ratios = jax.vmap(rotated_log_psi)(rotated_spinors) - log_psi
        overlap_kernel = jnp.exp(log_ratios)
        dimensions = 1 + jnp.asarray(self._twice_angular_momenta)
        weighted_characters = dimensions[:, None] * characters
        sector_weights = jnp.mean(
            weighted_characters * overlap_kernel[None, :], axis=-1
        )
        sector_energy_weights = (
            jnp.conj(prev_walker_stats["total_energy"]) * sector_weights
        )
        return {
            "angular_projection:weight": sector_weights,
            "angular_projection:energy_weight": sector_energy_weights,
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        """Average complex sector contributions without complex variances.

        Returns:
            Step-level ``angular_projection:weight`` and
            ``angular_projection:energy_weight``.
        """
        return mean_reduce(walker_stats, include_variance=False)

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: None
    ) -> dict[str, jnp.ndarray]:
        r"""Form sector energies from accumulated weights.

        Returns:
            Requested :math:`L` values with :math:`p_L` and
            :math:`E_L=h_L/p_L`.
        """
        del state
        sector_weights = jnp.nanmean(batched_stats["angular_projection:weight"], axis=0)
        sector_energy_weights = jnp.nanmean(
            batched_stats["angular_projection:energy_weight"], axis=0
        )
        sector_energies = sector_energy_weights / sector_weights
        angular_momenta = jnp.asarray(self.angular_momenta)

        return {
            "angular_projection:angular_momentum": angular_momenta,
            "angular_projection:weight": sector_weights,
            "angular_projection:energy": sector_energies,
            "angular_projection:energy_weight": sector_energy_weights,
        }
