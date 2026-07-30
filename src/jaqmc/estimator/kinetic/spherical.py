# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Kinetic energy estimator on a sphere with magnetic monopole.

Computes :math:`\Lambda^2 / (2R^2)` on a Haldane sphere with monopole
strength :math:`Q`.

.. seealso:: :doc:`/guide/estimators/kinetic` for the full formulation.
"""

from collections.abc import Mapping
from typing import Any, Literal

import jax
from jax import numpy as jnp
from jax.numpy import cos, sin, tan

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import with_imag, with_real
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


def _total_angular_momentum_operator_terms(electrons, Q: float):
    r"""Build the total-angular-momentum operator for one configuration.

    Total angular momentum measures how the wavefunction changes when every
    electron is rotated together. This helper returns the two geometric
    ingredients needed to apply its three Cartesian components:

    * ``directions[i, a]`` is the infinitesimal change in electron ``i``'s
      ``(theta, phi)`` coordinates under a rotation about axis ``a``.
      Differentiating the wavefunction along this direction gives the orbital
      rotation generator :math:`\mathcal{G}_a`.
    * ``monopole_terms[a]`` is the position-dependent correction
      :math:`M_a` caused by the monopole field.

    Together they define the physical total-angular-momentum component

    .. math::
        \hat L_a = -i\mathcal{G}_a + M_a.

    This helper does not evaluate angular momentum itself; it constructs the
    rotation fields used by :func:`_make_total_angular_momentum_estimator`.

    Returns:
        ``(directions, monopole_terms)`` for axes ordered ``(x, y, z)``.
        Their shapes are ``(n_electrons, 3, 2)`` and ``(3,)``, respectively.
    """
    theta, phi = electrons[..., 0], electrons[..., 1]
    r_hat = jnp.stack(
        [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
    )
    phi_hat = jnp.stack([-sin(phi), cos(phi), jnp.zeros_like(phi)], axis=-1)
    theta_hat_prime = jnp.stack(
        [cos(phi) / tan(theta), sin(phi) / tan(theta), -jnp.ones_like(theta)],
        axis=-1,
    )
    directions = jnp.stack([phi_hat, -theta_hat_prime], axis=-1)
    monopole_terms = Q * jnp.sum(
        theta_hat_prime * cos(theta)[..., None] + r_hat, axis=-2
    )
    return directions, monopole_terms


def _make_total_angular_momentum_estimator(f, Q: float):
    r"""Create a one-walker evaluator for total angular momentum.

    For each Cartesian axis :math:`a`, the operator is
    :math:`\hat L_a=-i\mathcal{G}_a+M_a`: :math:`\mathcal{G}_a` differentiates
    along the rotation along axis `a`, and :math:`M_a` is the monopole contribution.

    Nested JVPs compute :math:`\mathcal{G}_a\log\psi` and
    :math:`\mathcal{G}_a^2\log\psi`. Because the rotation direction depends on
    the electron coordinates, the outer JVP also differentiates the direction
    itself. These derivatives are combined as

    .. math::
        \frac{\hat L_a^2\psi}{\psi}
        = -\mathcal{G}_a^2\log\psi
          -(\mathcal{G}_a\log\psi)^2
          -2iM_a\mathcal{G}_a\log\psi
          -i\mathcal{G}_aM_a + M_a^2.

    Summing over the three axes gives :math:`L^2`. The z-axis has
    :math:`M_z=0` in this convention, so its derivatives also give
    :math:`L_z` and :math:`L_z^2`.

    Args:
        f: Function ``(params, electrons) -> complex scalar`` returning
            ``log(psi)`` for one walker.
        Q: Magnetic monopole strength, equal to half the flux.

    Returns:
        A function accepting ``(params, electrons)`` and returning the local
        ``angular_momentum_z``, ``angular_momentum_z_square``, and
        ``angular_momentum_square`` observables.
    """

    def evaluate(params, electrons):
        directions, _ = _total_angular_momentum_operator_terms(electrons, Q)

        def evaluate_component(component):
            def first_order_terms(x):
                rotation_directions, monopole_terms = (
                    _total_angular_momentum_operator_terms(x, Q)
                )
                rotation_direction = rotation_directions[..., component, :]
                g_log_psi = jax.jvp(
                    lambda y: f(params, y), (x,), (rotation_direction,)
                )[1]
                return g_log_psi, monopole_terms[component]

            rotation_direction = directions[..., component, :]
            ((g_log_psi, monopole_term), (g_squared_log_psi, monopole_derivative)) = (
                jax.jvp(first_order_terms, (electrons,), (rotation_direction,))
            )

            angular_momentum = -1j * g_log_psi + monopole_term
            angular_momentum_squared = (
                -g_squared_log_psi
                - g_log_psi**2
                - 2j * monopole_term * g_log_psi
                - 1j * monopole_derivative
                + monopole_term**2
            )
            return angular_momentum, angular_momentum_squared

        components, component_squares = zip(
            *(evaluate_component(component) for component in range(3)),
            strict=True,
        )

        return {
            "angular_momentum_z": components[2].real,
            "angular_momentum_z_square": component_squares[2].real,
            "angular_momentum_square": sum(component_squares, start=0.0j).real,
        }

    return evaluate


@configurable_dataclass
class SphericalKinetic(PerWalkerEstimator):
    r"""Local kinetic and total-angular-momentum estimators on a sphere.

    For each walker, this computes the kinetic energy
    :math:`\sum_i(\Lambda_i^2\psi/\psi)/(2R^2)` and reports local
    :math:`L_z`, :math:`L_z^2`, and :math:`L^2`.

    ``forward_laplacian`` avoids a full coordinate Hessian but evaluates the
    angular-momentum terms separately. ``hessian`` forms that Hessian and
    reuses it for all outputs. Both modes compute the same quantities.

    Coordinates must avoid the poles, where the spherical chart is singular.

    Args:
        mode: ``"forward_laplacian"`` or ``"hessian"``.
        monopole_strength: Monopole strength :math:`Q = \mathrm{flux}/2`.
        radius: Sphere radius. Defaults to :math:`\sqrt{Q}` for ``Q > 0``.
        f_log_psi: Complex log-psi function (runtime dep).
        data_field: Electron-coordinate field name.
    """

    mode: Literal["hessian", "forward_laplacian"] = "hessian"
    monopole_strength: float = 1.0
    radius: float | None = None
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
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
        if self.mode == "forward_laplacian":
            return self._evaluate_forward_laplacian(params, data, state)
        if self.mode == "hessian":
            return self._evaluate_hessian(params, data, state)
        raise ValueError(f"Unsupported Laplacian mode {self.mode}.")

    def _evaluate_hessian(
        self, params: Params, data: Data, state: None
    ) -> tuple[dict[str, Any], None]:
        Q = self.monopole_strength
        r = jnp.array(self.radius if self.radius is not None else jnp.sqrt(Q))
        electrons = data[self.data_field]

        def f(p, x):
            return self.f_log_psi(p, data.merge({self.data_field: x}))

        theta, phi = electrons[..., 0], electrons[..., 1]

        # First derivatives
        grad_real = jax.grad(with_real(f), argnums=1)(params, electrons)
        grad_imag = jax.grad(with_imag(f), argnums=1)(params, electrons)
        grad_theta = grad_real[..., 0] + 1j * grad_imag[..., 0]
        grad_phi = grad_real[..., 1] + 1j * grad_imag[..., 1]

        # |grad log psi|^2 on a sphere
        square_grad_logpsi = jnp.sum(grad_theta**2 + grad_phi**2 / sin(theta) ** 2)

        # Second derivatives (Hessian)
        hess_real = jax.hessian(with_real(f), argnums=1)(params, electrons)
        hess_imag = jax.hessian(with_imag(f), argnums=1)(params, electrons)
        hess_logpsi = hess_real + 1j * hess_imag

        # Spherical Laplacian of log psi
        grad_grad_logpsi = jnp.sum(
            grad_theta / tan(theta)
            + jnp.diagonal(hess_logpsi[:, 0, :, 0])
            + jnp.diagonal(hess_logpsi[:, 1, :, 1]) / sin(theta) ** 2
        )

        # Magnetic contribution (section 3.10.3 of "Composite Fermions")
        magnetic_contribution = jnp.sum(
            (Q / tan(theta)) ** 2 + 2j * Q * cos(theta) / sin(theta) ** 2 * grad_phi
        )
        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        kinetic_energy = sum_kinetic_momentum_square / 2 / r**2

        # Angular momentum from the Hessian
        i = (Ellipsis, slice(None), jnp.newaxis)
        j = (Ellipsis, jnp.newaxis, slice(None))
        r_hat = jnp.stack([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
        phi_hat = jnp.stack([-sin(phi), cos(phi), jnp.zeros_like(phi)])
        theta_hat_prime = jnp.stack(
            [
                cos(phi) / tan(theta),
                sin(phi) / tan(theta),
                -jnp.ones_like(theta),
            ]
        )

        hess_theta_theta = hess_logpsi[:, 0, :, 0] + grad_theta[*i] * grad_theta[*j]
        hess_theta_phi = hess_logpsi[:, 0, :, 1] + grad_theta[*i] * grad_phi[*j]
        hess_phi_phi = hess_logpsi[:, 1, :, 1] + grad_phi[*i] * grad_phi[*j]

        magnetic_term = Q * (theta_hat_prime * cos(theta) + r_hat)
        angular_momentum_square = jnp.sum(
            2 * phi_hat[*i] * theta_hat_prime[*j] * hess_theta_phi
            - phi_hat[*i] * phi_hat[*j] * hess_theta_theta
            - theta_hat_prime[*i] * theta_hat_prime[*j] * hess_phi_phi
            - (2j * magnetic_term[*j])
            * (phi_hat[*i] * grad_theta[*i] - theta_hat_prime[*i] * grad_phi[*i])
            + magnetic_term[*i] * magnetic_term[*j],
        ) - jnp.sum(grad_theta / tan(theta))

        return {
            "energy:kinetic": kinetic_energy,
            "angular_momentum_z": jnp.sum(grad_phi).imag,
            "angular_momentum_z_square": -jnp.sum(hess_phi_phi).real,
            "angular_momentum_square": angular_momentum_square.real,
        }, state

    def _evaluate_forward_laplacian(
        self, params: Params, data: Data, state: None
    ) -> tuple[dict[str, Any], None]:
        from jaqmc.laplacian import forward_laplacian, make_laplacian_input

        Q = self.monopole_strength
        r = jnp.array(self.radius if self.radius is not None else jnp.sqrt(Q))
        electrons = data[self.data_field]

        def f(p, x):
            return self.f_log_psi(p, data.merge({self.data_field: x}))

        theta = electrons[..., 0]

        # Forward Laplacian with spherical metric weights
        fwdlap_weights = jnp.stack([jnp.ones_like(theta), 1 / sin(theta)], axis=-1)
        input_laptuple = make_laplacian_input(
            electrons,
            weights=fwdlap_weights,
            sparse_axis=0,
        )
        fwdlap_output = forward_laplacian(f)(params, input_laptuple)
        grad_logpsi = (
            fwdlap_output.dense_jacobian.reshape(electrons.shape) / fwdlap_weights
        )
        grad_theta, grad_phi = grad_logpsi[..., 0], grad_logpsi[..., 1]

        square_grad_logpsi = jnp.sum(grad_theta**2 + grad_phi**2 / sin(theta) ** 2)
        grad_grad_logpsi = jnp.sum(grad_theta / tan(theta)) + fwdlap_output.laplacian

        magnetic_contribution = jnp.sum(
            (Q / tan(theta)) ** 2 + 2j * Q * cos(theta) / sin(theta) ** 2 * grad_phi
        )
        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        kinetic_energy = sum_kinetic_momentum_square / 2 / r**2

        angular_stats = _make_total_angular_momentum_estimator(f, Q)(params, electrons)

        return {
            "energy:kinetic": kinetic_energy,
            **angular_stats,
        }, state
