# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Kinetic energy estimator on a sphere with magnetic monopole.

Computes :math:`\Lambda^2 / (2R^2)` on a Haldane sphere with monopole
strength :math:`Q`.

.. seealso:: :doc:`/guide/estimators/kinetic` for the full formulation.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp
from jax.numpy import cos, sin, tan

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import LocalEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import with_imag, with_real
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

from ._common import LaplacianMode


def _angular_momentum_operator(f, Q: float):
    r"""Create angular momentum operator :math:`\hat L`.

    Following section 3.10 of *Composite Fermions* (Jain):

    .. math::
        \hat L = -i\,\hat\phi\,\partial_\theta
                 + \hat\theta'\,i\,\partial_\phi
                 + Q(\cot\theta\,\hat\theta + \hat r)

    Returns the differential and magnetic terms separately so that
    :math:`L^2` can be assembled without a full Hessian.

    Args:
        f: Function ``(params, data) -> complex scalar``.
        Q: Monopole strength.

    Returns:
        A callable ``(params, data) -> (differential_term, magnetic_term)``
        where each term has shape ``(3,)``.
    """
    jac_f_real = jax.jacrev(lambda p, d: f(p, d).real, argnums=1)
    jac_f_imag = jax.jacrev(lambda p, d: f(p, d).imag, argnums=1)

    def operator(params, data):
        theta, phi = data[..., 0], data[..., 1]
        jacobian = jac_f_real(params, data) + 1j * jac_f_imag(params, data)
        grad_theta, grad_phi = jacobian[..., 0, None], jacobian[..., 1, None]

        r_hat = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        phi_hat = jnp.stack([-sin(phi), cos(phi), jnp.zeros_like(phi)], axis=-1)
        theta_hat_prime = jnp.stack(
            [cos(phi) / tan(theta), sin(phi) / tan(theta), -jnp.ones_like(theta)],
            axis=-1,
        )

        differential_term = jnp.sum(
            -1j * (phi_hat * grad_theta - theta_hat_prime * grad_phi), axis=-2
        )
        magnetic_term = Q * jnp.sum(
            theta_hat_prime * cos(theta)[:, None] + r_hat, axis=0
        )
        return differential_term, magnetic_term

    return operator


def _angular_momentum_square(f, Q: float):
    r"""Compute :math:`(\hat L^2\psi)/\psi` via double operator application.

    Applies :math:`\hat L` twice to avoid a full Hessian. See the derivation
    in ``deephall/hamiltonian.py`` for how the cross-terms simplify.

    Args:
        f: Function ``(params, data) -> complex scalar`` (log-psi).
        Q: Monopole strength.

    Returns:
        A callable ``(params, data) -> dict`` with angular momentum stats.
    """
    L_on_logpsi = _angular_momentum_operator(f, Q)
    L_squared_on_logpsi = _angular_momentum_operator(
        lambda p, d: sum(L_on_logpsi(p, d)), Q
    )

    def compute(params, data):
        angular_momentum = sum(L_on_logpsi(params, data))
        l2_components = (
            jnp.diag(L_squared_on_logpsi(params, data)[0]) + angular_momentum**2
        )
        return {
            "angular_momentum_z": angular_momentum[2].real,
            "angular_momentum_z_square": l2_components[2].real,
            "angular_momentum_square": jnp.sum(l2_components).real,
        }

    return compute


@configurable_dataclass
class SphericalKinetic(LocalEstimator):
    r"""Kinetic energy on a sphere with magnetic monopole.

    Uses the Hessian-based calculation following the formulas in
    section 3.10.3 of *Composite Fermions* (Jain):

    .. math::
        \frac{|\Lambda|^2 \psi}{2R^2 \psi}
        = \frac{1}{2R^2}\left[
            -R^2 \frac{\nabla^2\psi}{\psi}
            + (Q\cot\theta)^2
            + 2iQ \frac{\cot\theta}{\sin\theta}
              \frac{\partial\log\psi}{\partial\phi}
        \right]

    Also computes angular momentum observables.

    Args:
        mode: Laplacian computation mode. ``scan`` and ``fori_loop`` use a
            Hessian-based approach; ``forward_laplacian`` uses the forward
            Laplacian.
        monopole_strength: Half the magnetic flux (:math:`Q = \text{flux}/2`).
        radius: Sphere radius. Defaults to :math:`\sqrt{Q}`.
        f_log_psi: Complex log-psi function (runtime dep).
        data_field: Name of the data field (runtime dep, default ``"electrons"``).
    """

    mode: LaplacianMode = LaplacianMode.scan
    monopole_strength: float = 1.0
    radius: float | None = None
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

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
        return self._evaluate_hessian(params, data, state)

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
        from folx import forward_laplacian

        Q = self.monopole_strength
        r = jnp.array(self.radius if self.radius is not None else jnp.sqrt(Q))
        electrons = data[self.data_field]

        def f(p, x):
            return self.f_log_psi(p, data.merge({self.data_field: x}))

        theta = electrons[..., 0]

        # Forward Laplacian with spherical metric weights
        fwd_f = forward_laplacian(lambda x: f(params, x))
        fwdlap_weights = jnp.stack([jnp.ones_like(theta), 1 / sin(theta)], axis=-1)
        fwdlap_output = fwd_f(electrons, weights=fwdlap_weights)
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

        # Angular momentum via double operator application
        angular_stats = _angular_momentum_square(f, Q)(params, electrons)

        return {
            "energy:kinetic": kinetic_energy,
            **angular_stats,
        }, state
