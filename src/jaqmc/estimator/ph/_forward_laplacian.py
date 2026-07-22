# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Forward Laplacian backend for the PH derivative term — production path.

Computes the PH derivative term in a single Forward Laplacian pass by
folding the per-electron Cholesky factor of the mass matrix into a
shifted-coordinate map, following Fu et al. (2025). The math reference
implementation lives in :mod:`jaqmc.estimator.ph._standard`.

.. seealso:: :ref:`ph-backend-fl` for the Cholesky derivation and
   :ref:`ph-formulas` for the underlying mass matrix and first-order
   vector definitions.

References:
    Bennett, M. C., Reboredo, F. A., Mitas, L., Krogel, J. T.,
    "High Accuracy Transition Metal Effective Cores for the Many-Body
    Diffusion Monte Carlo Method," *J. Chem. Theory Comput.* **18**,
    828-839 (2022). DOI: `10.1021/acs.jctc.1c00992
    <https://doi.org/10.1021/acs.jctc.1c00992>`_ — PH derivative
    formulation.

    Fu, W. *et al.*, "Local Pseudopotential Unlocks the True Potential
    of Neural Network-based Quantum Monte Carlo," arXiv:2505.19909
    (2025) — NNQMC + PH + Forward Laplacian integration that this
    backend implements.
"""

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params
from jaqmc.data import Data
from jaqmc.utils.func_transform import grad_maybe_complex
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

__all__ = ["compute_derivative_energy"]


def compute_derivative_energy(
    f_log_psi: NumericWavefunctionEvaluate,
    params: Params,
    data: Data,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
    *,
    electrons_field: str,
) -> jnp.ndarray:
    r"""Evaluate the PH derivative term via Cholesky + forward Laplacian.

    Args:
        f_log_psi: Log-wavefunction evaluate function.
        params: Wavefunction parameters.
        data: Single-walker data. Electron positions are taken from
            ``data[electrons_field]`` and that field is replaced during the
            Forward Laplacian pass.
        ph_atoms: PH atom positions with shape ``(n_ph_atoms, 3)``.
        l2_values: Radial values :math:`\ell_2(r) = r\,v_{L^2}(r)` with shape
            ``(n_electrons, n_ph_atoms)``.
        electrons_field: Name of the electron-coordinate field in ``data``.

    Returns:
        Scalar PH derivative energy :math:`\sum_i E_{\mathrm{PH}, i}`.

    Raises:
        ValueError: If ``data[electrons_field]``, ``ph_atoms``, or
            ``l2_values`` do not have the expected shapes.

    .. seealso:: :ref:`ph-backend-fl` for the rendered Cholesky derivation.
    """
    electrons = data[electrons_field]
    if electrons.ndim != 2 or electrons.shape[-1] != 3:
        raise ValueError(f"data[{electrons_field!r}] must have shape (n_electrons, 3)")
    if ph_atoms.ndim != 2 or ph_atoms.shape[-1] != 3:
        raise ValueError("ph_atoms must have shape (n_ph_atoms, 3)")
    if l2_values.shape != (electrons.shape[0], ph_atoms.shape[0]):
        raise ValueError("l2_values must have shape (n_electrons, n_ph_atoms)")

    mass = jax.vmap(_mass_matrix_for_electron, in_axes=(0, None, 0))(
        electrons, ph_atoms, l2_values
    )
    cholesky = jnp.linalg.cholesky(mass)

    second_order = _second_order_term(
        f_log_psi=f_log_psi,
        params=params,
        data=data,
        electrons=electrons,
        cholesky=cholesky,
        electrons_field=electrons_field,
    )
    grad_logpsi = _logpsi_gradient(
        f_log_psi=f_log_psi,
        params=params,
        data=data,
        electrons=electrons,
        electrons_field=electrons_field,
    )
    first_order_vectors = jax.vmap(
        _first_order_vector_for_electron, in_axes=(0, None, 0)
    )(electrons, ph_atoms, l2_values)
    first_order_term = jnp.sum(first_order_vectors * grad_logpsi)

    return second_order + first_order_term


def _mass_matrix_for_electron(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_row: jnp.ndarray,
) -> jnp.ndarray:
    r"""Assemble the per-electron mass matrix.

    Implements the Bennett et al. (2022) PH mass matrix in the XML
    :math:`\ell_2 = r\,v_{L^2}` convention,

    .. math::

        M = \tfrac{1}{2} I + \sum_a \ell_2(r_a)
            \left(r_a I - \frac{r_a r_a^\top}{|r_a|}\right).

    A non-PD assembly surfaces as NaN downstream via
    :func:`jnp.linalg.cholesky`, matching the loud failure mode adopted
    by this backend (Fu et al. 2025).

    Returns:
        A ``(3, 3)`` mass matrix.

    .. seealso:: :ref:`ph-mass-matrix` for the rendered derivation.
    """
    rel = electron[None, :] - ph_atoms
    r = jnp.linalg.norm(rel, axis=-1)
    outer = rel[:, :, None] * rel[:, None, :]
    eye = jnp.eye(3, dtype=electron.dtype)
    inv_r = jnp.where(r > 0, 1.0 / r, 0.0)
    return 0.5 * eye + jnp.sum(
        l2_row[:, None, None] * (r[:, None, None] * eye - outer * inv_r[:, None, None]),
        axis=0,
    )


def _first_order_vector_for_electron(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_row: jnp.ndarray,
) -> jnp.ndarray:
    r"""Analytic first-order PH vector.

    Bennett et al. (2022) gives the divergence of the L2 part of the mass
    tensor as

    .. math::

        b_j = -\partial_i \left[ v_{L^2}(r) (r^2 \delta_{ij} - r_i r_j) \right]
            = 2\, v_{L^2}(r)\, r_j,

    which in the XML :math:`\ell_2 = r\,v_{L^2}` convention becomes
    :math:`b = \sum_a 2\,\ell_2(r_a)/r_a \cdot (x - R_a)`.

    Returns:
        A ``(3,)`` first-order vector.

    .. seealso:: :ref:`ph-first-order-vector` for the rendered derivation.
    """
    rel = electron[None, :] - ph_atoms
    r = jnp.linalg.norm(rel, axis=-1)
    inv_r = jnp.where(r > 0, 1.0 / r, 0.0)
    return jnp.sum(2.0 * l2_row[:, None] * inv_r[:, None] * rel, axis=0)


def _second_order_term(
    *,
    f_log_psi: NumericWavefunctionEvaluate,
    params: Params,
    data: Data,
    electrons: jnp.ndarray,
    cholesky: jnp.ndarray,
    electrons_field: str,
) -> jnp.ndarray:
    r"""Compute :math:`-\mathrm{Tr}(M H) - g^\top M g` summed over electrons.

    This is the Fu et al. (2025) Cholesky-shifted Forward Laplacian
    construction. With :math:`M = L L^\top` per electron, define

    .. math::

        g(y) = \log\psi(\dots, x_e + L_e y_e, \dots)
        \quad\text{at}\quad y = 0.

    Then :math:`\nabla_y g = L^\top \nabla_x \log\psi` and
    :math:`\mathrm{Tr}(\nabla_y^2 g) = \mathrm{Tr}(L^\top H L) =
    \mathrm{Tr}(M H)`, so a single ``forward_laplacian`` pass over
    :math:`g` yields both required quantities.

    Returns:
        Scalar :math:`-\mathrm{Tr}(M H) - g^\top M g`.

    .. seealso:: :ref:`ph-backend-fl` for the rendered derivation.
    """
    from jaqmc.laplacian import forward_laplacian, make_laplacian_input

    def shifted_logpsi(y: jnp.ndarray) -> jnp.ndarray:
        shift = jnp.einsum("eij,ej->ei", cholesky, y)
        return f_log_psi(params, data.merge({electrons_field: electrons + shift}))

    result = forward_laplacian(shifted_logpsi)(
        make_laplacian_input(jnp.zeros_like(electrons), sparse_axis=0)
    )
    grad_y = result.dense_jacobian
    # NOTE: jnp.vdot conjugates its first argument, so it returns
    # Σ |g_i|² rather than Σ g_i². The two agree for real `log psi`, which
    # is the only case currently supported by this backend (matching the
    # caveat in the PHEnergy class docstring). Switch to
    # jnp.sum(grad_y * grad_y) before enabling complex wavefunctions on PH.
    return -(result.laplacian + jnp.vdot(grad_y, grad_y))


def _logpsi_gradient(
    *,
    f_log_psi: NumericWavefunctionEvaluate,
    params: Params,
    data: Data,
    electrons: jnp.ndarray,
    electrons_field: str,
) -> jnp.ndarray:
    r"""Return :math:`\nabla_x \log\psi` with shape ``(n_electrons, 3)``."""
    electron_shape = electrons.shape

    def logpsi_flat(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(electron_shape)
        return f_log_psi(params, data.merge({electrons_field: x}))

    grad_flat = grad_maybe_complex(logpsi_flat)(electrons.reshape(-1))
    return grad_flat.reshape(electron_shape)
