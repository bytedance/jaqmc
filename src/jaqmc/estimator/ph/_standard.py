# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Standard PH derivative backend — reverse-mode reference implementation.

This file is the paper-faithful math anchor for the PH derivative term;
performance and pathological-input hardening live in
:mod:`jaqmc.estimator.ph._forward_laplacian`, the production path. The two
backends are cross-checked by parity tests in
``tests/estimator/ph/ph_estimator_parity_test.py``.

.. seealso:: :ref:`ph-backend-standard` for the algorithm overview and
   :ref:`ph-formulas` for the underlying mass matrix and first-order
   vector definitions.

References:
    Bennett, M. C., Reboredo, F. A., Mitas, L., Krogel, J. T.,
    "High Accuracy Transition Metal Effective Cores for the Many-Body
    Diffusion Monte Carlo Method," *J. Chem. Theory Comput.* **18**,
    828-839 (2022). DOI: `10.1021/acs.jctc.1c00992
    <https://doi.org/10.1021/acs.jctc.1c00992>`_.
"""

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params
from jaqmc.data import Data
from jaqmc.utils.func_transform import grad_maybe_complex, hessian_maybe_complex
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

__all__ = [
    "build_second_order_matrix",
    "compute_derivative_energy",
    "evaluate_ph_first_order_vector",
]


def compute_derivative_energy(
    f_log_psi: NumericWavefunctionEvaluate,
    params: Params,
    data: Data,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
    *,
    electrons_field: str,
) -> jnp.ndarray:
    r"""Evaluate the PH derivative term from its operator definition.

    Assembles the per-electron PH derivative term directly from the
    Bennett et al. (2022) operator decomposition

    .. math::

        E_{\mathrm{PH}, i}
        = -\mathrm{Tr}(M_i H_i) - g_i^\top M_i g_i + b_i^\top g_i,

    using :func:`jax.grad` for :math:`g = \nabla \log \psi`,
    :func:`jax.hessian` for the per-electron Hessian block :math:`H`, the
    operator-form mass matrix :math:`M` from
    :func:`build_second_order_matrix`, and the analytic first-order vector
    :math:`b` from :func:`evaluate_ph_first_order_vector`. The two einsums
    contract :math:`M` against :math:`H` (trace term) and against :math:`g`
    on both sides (quadratic term); the third term reduces along the
    first-order vector. Total cost: one ``jax.grad`` pass for :math:`g`,
    one ``jax.hessian`` pass for :math:`H`, and one ``jax.jacfwd`` per
    electron inside ``evaluate_ph_first_order_vector``.

    This is the educational reference path. The production ``forward_laplacian``
    backend (:mod:`jaqmc.estimator.ph._forward_laplacian`) evaluates the
    same quantity through a Cholesky-shifted Forward Laplacian pass.

    Args:
        f_log_psi: Log-wavefunction evaluate function.
        params: Wavefunction parameters.
        data: Single-walker data. Electron positions are taken from
            ``data[electrons_field]`` and that field is substituted while
            differentiating.
        ph_atoms: PH atom positions with shape ``(n_ph_atoms, 3)``.
        l2_values: Radial values :math:`\ell_2(r) = r\,v_{L^2}(r)` with
            shape ``(n_electrons, n_ph_atoms)``.
        electrons_field: Name of the electron-coordinate field in ``data``.

    Returns:
        Scalar PH derivative energy :math:`\sum_i E_{\mathrm{PH}, i}`.

    Raises:
        ValueError: If ``data[electrons_field]``, ``ph_atoms``, or
            ``l2_values`` do not have the expected shapes.

    .. seealso:: :ref:`ph-backend-standard` for the rendered algorithm
       overview.
    """
    electrons = data[electrons_field]
    if electrons.ndim != 2 or electrons.shape[-1] != 3:
        raise ValueError(f"data[{electrons_field!r}] must have shape (n_electrons, 3)")
    if ph_atoms.ndim != 2 or ph_atoms.shape[-1] != 3:
        raise ValueError("ph_atoms must have shape (n_ph_atoms, 3)")
    if l2_values.shape != (electrons.shape[0], ph_atoms.shape[0]):
        raise ValueError("l2_values must have shape (n_electrons, n_ph_atoms)")

    grad_blocks = _logpsi_gradient(
        f_log_psi=f_log_psi,
        params=params,
        data=data,
        electrons=electrons,
        electrons_field=electrons_field,
    )
    hessian_blocks = _logpsi_hessian_blocks(
        f_log_psi=f_log_psi,
        params=params,
        data=data,
        electrons=electrons,
        electrons_field=electrons_field,
    )
    full_mass = _build_mass_matrices(electrons, ph_atoms, l2_values)
    trace_term = jnp.einsum("eij,eji->", full_mass, hessian_blocks)
    grad_term = jnp.einsum("ei,eij,ej->", grad_blocks, full_mass, grad_blocks)
    first_order_term = _first_order_term(electrons, ph_atoms, l2_values, grad_blocks)
    return -(trace_term + grad_term) + first_order_term


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


def _logpsi_hessian_blocks(
    *,
    f_log_psi: NumericWavefunctionEvaluate,
    params: Params,
    data: Data,
    electrons: jnp.ndarray,
    electrons_field: str,
) -> jnp.ndarray:
    r"""Return per-electron Hessian blocks of :math:`\log\psi`.

    Output shape ``(n_electrons, 3, 3)``: the per-electron diagonal blocks
    of the full Hessian (cross-electron blocks are discarded since they do
    not enter the PH derivative term).
    """
    flat_electrons = electrons.reshape(-1)
    electron_shape = electrons.shape

    def logpsi_flat(x: jnp.ndarray) -> jnp.ndarray:
        reshaped = jnp.reshape(x, electron_shape)
        return f_log_psi(params, data.merge({electrons_field: reshaped}))

    hessian_flat = hessian_maybe_complex(logpsi_flat)(flat_electrons)
    n_electrons = electrons.shape[0]
    return hessian_flat.reshape(n_electrons, 3, n_electrons, 3)[
        jnp.arange(n_electrons), :, jnp.arange(n_electrons), :
    ]


def _build_mass_matrices(
    electrons: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
) -> jnp.ndarray:
    r"""Return per-electron mass matrices stacked along axis 0 (shape ``(N, 3, 3)``)."""
    return jax.vmap(
        lambda electron, l2_row: build_second_order_matrix(electron, ph_atoms, l2_row),
        in_axes=(0, 0),
    )(electrons, l2_values)


def _first_order_term(
    electrons: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
    grad_blocks: jnp.ndarray,
) -> jnp.ndarray:
    r"""Return :math:`\sum_i b_i^\top g_i`, the first-order PH energy contribution."""
    first_order_vectors = jax.vmap(
        lambda electron, l2_row: evaluate_ph_first_order_vector(
            electron, ph_atoms, l2_row
        ),
        in_axes=(0, 0),
    )(electrons, l2_values)
    return jnp.einsum("ei,ei->", grad_blocks, first_order_vectors)


def build_second_order_matrix(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
) -> jnp.ndarray:
    r"""Build the PH second-order mass matrix for one electron.

    Starting from the Bennett et al. (2022) PH diffusion tensor

    .. math::

        d(r_{iI}) = I + 2 \left(r_{iI}^2 I - r_{iI} r_{iI}^\top \right)
        v_{L^2}^I(r_{iI}),

    the per-electron mass tensor is :math:`M = \frac{1}{2} d`. Using the XML
    convention ``l2 = r v_{L^2}``, this becomes

    .. math::

        M = \frac{1}{2} I + \sum_a \ell_2(r_a)
            \left(r_a I - \frac{r_a r_a^T}{|r_a|}\right),

    where :math:`r_a = x - R_a` is the electron-atom displacement.

    The assembled :math:`M` is returned as-is when it is positive definite
    and as a NaN-filled matrix otherwise, so non-PD assembly surfaces as NaN
    downstream uniformly across both PH backends (the ``forward_laplacian``
    backend reaches the same NaN state via :func:`jnp.linalg.cholesky`).

    Args:
        electron: Electron position with shape ``(3,)``.
        ph_atoms: PH atom positions with shape ``(m, 3)``.
        l2_values: Radial :math:`\ell_2(r)` values with shape ``(m,)``.

    Returns:
        A ``(3, 3)`` mass matrix for a single electron.

    .. seealso:: :ref:`ph-mass-matrix` for the rendered derivation.
    """
    electron = jnp.asarray(electron)
    ph_atoms = jnp.asarray(ph_atoms)
    l2_values = jnp.asarray(l2_values)
    _validate_ph_geometry(electron, ph_atoms, l2_values, "l2_values")

    return _build_second_order_matrix_single(electron, ph_atoms, l2_values)


def evaluate_ph_first_order_vector(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
) -> jnp.ndarray:
    r"""Evaluate the PH first-order vector field for one electron.

    The Bennett et al. (2022) differential form implies

    .. math::

        b = - \nabla \cdot \left(M - \tfrac{1}{2} I\right).

    We evaluate the divergence of the same second-order matrix used in
    local-energy evaluation by the ``standard`` backend. The
    ``forward_laplacian`` backend uses an analytic divergence directly and
    does not call this helper.

    Args:
        electron: Electron position with shape ``(3,)``.
        ph_atoms: PH atom positions with shape ``(m, 3)``.
        l2_values: Radial :math:`\ell_2(r)` values with shape ``(m,)``.

    Returns:
        A vector with shape ``(3,)``.

    .. seealso:: :ref:`ph-first-order-vector` for the rendered derivation.
    """
    electron = jnp.asarray(electron)
    ph_atoms = jnp.asarray(ph_atoms)
    l2_values = jnp.asarray(l2_values)
    _validate_ph_geometry(electron, ph_atoms, l2_values, "l2_values")

    return _evaluate_ph_first_order_vector_single(electron, ph_atoms, l2_values)


def _validate_ph_geometry(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    radial_values: jnp.ndarray,
    radial_name: str,
) -> None:
    if electron.shape != (3,):
        raise ValueError("electron must have shape (3,)")
    if ph_atoms.ndim != 2 or ph_atoms.shape[-1] != 3:
        raise ValueError("ph_atoms must have shape (m, 3)")
    if radial_values.ndim != 1:
        raise ValueError(f"{radial_name} must be 1-D with shape (m,)")
    if radial_values.shape[0] != ph_atoms.shape[0]:
        raise ValueError(f"{radial_name} must have shape (m,) matching ph_atoms")


def _build_second_order_matrix_single(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
) -> jnp.ndarray:
    # Bennett et al. (2022), eq. for the PH diffusion tensor:
    #     d(r) = I + 2 (r^2 I - r r^T) v_L2(r),
    # with M = d/2. In the XML r*V convention l2(r) = r * v_L2(r), so
    #     M = 0.5 I + sum_a l2(r_a) * (r_a I - r_a r_a^T / |r_a|).
    rel = electron[None, :] - ph_atoms
    r = jnp.linalg.norm(rel, axis=-1)
    outer = rel[:, :, None] * rel[:, None, :]
    eye = jnp.eye(3, dtype=electron.dtype)
    inv_r = jnp.where(r > 0, 1.0 / r, 0.0)
    mass = jnp.sum(
        l2_values[:, None, None]
        * (r[:, None, None] * eye - outer * inv_r[:, None, None]),
        axis=0,
    )
    full = mass + 0.5 * eye
    # Non-PD assembly surfaces as NaN downstream uniformly across both
    # backends (forward_laplacian reaches the same state via cholesky).
    return jnp.where(
        jnp.all(jnp.linalg.eigvalsh(full) > 0.0),
        full,
        jnp.full_like(full, jnp.nan),
    )


def _evaluate_ph_first_order_vector_single(
    electron: jnp.ndarray,
    ph_atoms: jnp.ndarray,
    l2_values: jnp.ndarray,
) -> jnp.ndarray:
    # Bennett et al. (2022): b = -div(M - I/2). We compute this divergence
    # numerically via jacfwd over the same correction matrix `M - I/2` used
    # in the second-order term, so NaN propagation from non-PD `M` at the
    # evaluation point flows naturally into the divergence. The
    # `forward_laplacian` backend instead uses the analytic identity
    #     b_j = sum_a 2 v_L2(r_a) (x - R_a)_j
    # in the same XML convention; the two paths are cross-checked in
    # tests/estimator/ph/ph_standard_test.py.
    eye = jnp.eye(3, dtype=electron.dtype)

    def correction_matrix(position: jnp.ndarray) -> jnp.ndarray:
        return (
            _build_second_order_matrix_single(position, ph_atoms, l2_values) - 0.5 * eye
        )

    jacobian = jax.jacfwd(correction_matrix)(electron)
    return -jnp.einsum("jkj->k", jacobian)
