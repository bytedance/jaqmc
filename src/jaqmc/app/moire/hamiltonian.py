# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Mapping
from typing import Any, cast

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.estimator.ewald import EwaldSum2D
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import linearize_maybe_complex
from jaqmc.utils.supercell import get_reciprocal_vectors
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.output.logdet import ComplexLogDetOutput, LogDet

from .data import MoireData


def _moire_first_shell_vectors(primitive_lattice: jnp.ndarray) -> jnp.ndarray:
    r"""Returns the six first-shell moire reciprocal vectors.

    For the canonical hexagonal moire lattice these correspond to

    .. math::

        \mathbf g_i = \frac{4\pi}{\sqrt{3}a_M}
        \left(\cos\frac{\pi(i-1)}{3}, \sin\frac{\pi(i-1)}{3}\right),
        \quad i=1,\ldots,6.

    The implementation uses the reciprocal vectors of ``primitive_lattice``
    directly, with the index order ``[b1, b2, b2-b1, -b1, -b2, b1-b2]``.

    Args:
        primitive_lattice: Direct primitive-cell lattice vectors with shape
            ``(2, 2)``.

    Returns:
        Cartesian reciprocal vectors :math:`\mathbf g_i` with shape ``(6, 2)``.
    """
    b1, b2 = get_reciprocal_vectors(jnp.asarray(primitive_lattice))
    return jnp.stack([b1, b2, b2 - b1, -b1, -b2, b1 - b2], axis=0)


def _moire_valley_momenta(
    primitive_lattice: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Returns the valley momenta :math:`\mathbf K_+` and :math:`\mathbf K_-`.

    The code uses

    .. math::

        \mathbf K_+ = (\mathbf b_1 + \mathbf b_2)/3,\qquad
        \mathbf K_- = (2\mathbf b_1 - \mathbf b_2)/3.

    Args:
        primitive_lattice: Direct primitive-cell lattice vectors with shape
            ``(2, 2)``.

    Returns:
        A tuple ``(K_plus, K_minus)`` of Cartesian valley momenta.
    """
    b1, b2 = get_reciprocal_vectors(jnp.asarray(primitive_lattice))
    return (b1 + b2) / 3.0, (2.0 * b1 - b2) / 3.0


def _physical_spin_signs(nspins: tuple[int, int], dtype: jnp.dtype) -> jnp.ndarray:
    """Returns physical-spin signs for all electrons.

    Args:
        nspins: Tuple of ``(n_up, n_down)`` electrons.
        dtype: Floating dtype of the returned signs.

    Returns:
        Array containing ``+1`` for spin-up electrons and ``-1`` for spin-down
        electrons.
    """
    return jnp.concatenate(
        [
            jnp.ones((nspins[0],), dtype=dtype),
            -jnp.ones((nspins[1],), dtype=dtype),
        ]
    )


def _orbitals_to_det_weights(
    base_orbitals: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns relative determinant weights and orbital inverses."""
    logdet_out = cast(ComplexLogDetOutput, LogDet().apply({}, base_orbitals))
    det_weights = logdet_out["sign_logdets"] * jnp.exp(
        logdet_out["abs_logdets"] - logdet_out["logpsi"][..., None]
    )
    return det_weights, jnp.linalg.inv(base_orbitals)


def compute_operator_ratio(
    base_orbitals: jnp.ndarray,
    operator_orbitals: jnp.ndarray,
) -> jnp.ndarray:
    r"""Computes the local per-electron ratio of a one-body operator.

    Args:
        base_orbitals: Identity orbitals with shape ``(ndets, nelec, nelec)``.
        operator_orbitals: Operator-transformed orbitals with the same trailing
            three axes and optional leading operator axes.

    Returns:
        Per-electron ratios with shape ``(..., nelec)``.
    """
    det_weights, orbital_inv = _orbitals_to_det_weights(base_orbitals)
    determinant_ratios = jnp.einsum("...dij,dji->...di", operator_orbitals, orbital_inv)
    return jnp.sum(determinant_ratios * det_weights[..., None], axis=-2)


def _build_exchanged_orbitals(
    *,
    base_orbitals: jnp.ndarray,
    operator_orbitals: jnp.ndarray,
    exchange_masks: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Builds row-exchanged matrices for every operator and electron.

    Returns:
        Row-exchanged matrices with operator and replaced-electron axes.
    """
    replace_row, keep_other_rows = exchange_masks
    exchanged = (
        operator_orbitals[:, :, None, :, :] * replace_row[None, None, ...]
        + base_orbitals[None, :, None, :, :] * keep_other_rows[None, None, ...]
    )
    return jnp.transpose(exchanged, (0, 1, 3, 4, 2))


def _build_exchanged_inverse(
    *,
    orbital_inv: jnp.ndarray,
    orbitals_diff: jnp.ndarray,
) -> jnp.ndarray:
    r"""Inverts base matrices with one operator-transformed row.

    Replacing row ``a`` of :math:`\Phi_0` with the corresponding row of an
    operator-transformed matrix is a rank-one update. Sherman-Morrison gives the
    updated inverse for every operator and replaced electron without fresh
    ``O(n^3)`` solves.

    Args:
        orbital_inv: Inverse base orbitals shaped ``(ndet, nelec, nelec)``.
        orbitals_diff: Operator/base differences shaped
            ``(noperator, ndet, nelec, nelec)``.

    Returns:
        Updated inverses shaped
        ``(noperator, ndet, nelec, nelec, nelec)``; the final axis indexes the
        replaced electron.
    """
    denominator = 1.0 + jnp.einsum("odak,dka->oda", orbitals_diff, orbital_inv)
    return (
        orbital_inv[None, ..., None]
        - jnp.einsum("dia,odak,dkl->odila", orbital_inv, orbitals_diff, orbital_inv)
        / denominator[:, :, None, None, :]
    )


def compute_operator_derivative_ratio(
    params: Params,
    data: MoireData,
    *,
    operator_orbitals_fn: Callable[
        [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray]
    ],
) -> jnp.ndarray:
    r"""Computes :math:`\nabla_i(O_i\Psi)/\Psi` for one-body operators.

    The network is linearized once for the full operator stack. Row-replacement
    inverses are then reused for every spatial coordinate. This differentiates
    the transformed numerator, not its local ratio.

    Args:
        params: Wavefunction parameters.
        data: Single-walker moire data.
        operator_orbitals_fn: Function returning base orbitals and transformed
            orbitals shaped ``(noperator, ndet, nelec, nelec)``.

    Returns:
        Derivative ratios shaped ``(noperator, nelec, ndim)``.
    """
    nelec, ndim = data.positions.shape
    ncoord = nelec * ndim
    eye = jnp.eye(nelec)
    ones = jnp.ones((nelec, nelec))
    replace_row = jax.vmap(jnp.outer)(eye, ones)
    exchange_masks = (replace_row, ones - replace_row)
    positions_flat = data.positions.reshape(-1)

    def orbitals_from_spatial(spatial_flat: jnp.ndarray) -> jnp.ndarray:
        positions = spatial_flat.reshape(nelec, ndim)
        base, transformed = operator_orbitals_fn(
            params, data.merge({"positions": positions})
        )
        return jnp.concatenate([base[None, ...], transformed], axis=0)

    orbitals, pushforward = linearize_maybe_complex(
        orbitals_from_spatial, positions_flat
    )
    base_orbitals, operator_orbitals = orbitals[0], orbitals[1:]
    det_weights, orbital_inv = _orbitals_to_det_weights(base_orbitals)
    determinant_ratios = (
        jnp.einsum("odij,dji->odi", operator_orbitals, orbital_inv)
        * det_weights[None, :, None]
    )
    exchanged_inv = _build_exchanged_inverse(
        orbital_inv=orbital_inv,
        orbitals_diff=operator_orbitals - base_orbitals[None, ...],
    )
    derivative_det = parallel_jax.pvary(
        jnp.zeros(
            (operator_orbitals.shape[0], base_orbitals.shape[0], nelec, ndim),
            dtype=orbitals.dtype,
        )
    )

    def body(acc: jnp.ndarray, coord_idx: jnp.ndarray):
        tangent = pushforward(
            parallel_jax.pvary(
                jax.nn.one_hot(coord_idx, ncoord, dtype=positions_flat.dtype)
            )
        )
        exchanged_tangent = _build_exchanged_orbitals(
            base_orbitals=tangent[0],
            operator_orbitals=tangent[1:],
            exchange_masks=exchange_masks,
        )
        contracted = jnp.einsum("odije,odjie->ode", exchanged_tangent, exchanged_inv)
        electron_idx = coord_idx // ndim
        dim_idx = coord_idx % ndim
        acc = acc.at[:, :, electron_idx, dim_idx].set(contracted[:, :, electron_idx])
        return acc, None

    derivative_det, _ = jax.lax.scan(
        body, derivative_det, jnp.arange(ncoord, dtype=jnp.int32)
    )
    return jnp.sum(derivative_det * determinant_ratios[..., None], axis=1)


@configurable_dataclass
class CoulombInteractionEnergy(PerWalkerEstimator):
    r"""Estimator for the electron-electron Coulomb term in Eq. (2).

    The paper writes this contribution as

    .. math::

        E_{\rm Coulomb}
        = \frac{1}{2}\sum_{i\ne j}v_E(\mathbf r_i-\mathbf r_j),

    with a uniform neutralizing background and Ewald summation.  ``EwaldSum2D``
    evaluates the dimensionless periodic 2D interaction for charge ``-1``
    electrons; this estimator multiplies by ``coulomb_prefactor_mev``.

    Args:
        supercell_lattice: Direct lattice vectors of the simulation supercell.
        coulomb_prefactor_mev: Coulomb-energy prefactor in meV.
        output_key: Output statistics key.
    """

    supercell_lattice: jnp.ndarray = runtime_dep()
    coulomb_prefactor_mev: float = runtime_dep(default=1.0)
    output_key: str = "energy:coulomb"

    def init(self, data: MoireData, rngs: PRNGKey) -> None:
        del data, rngs
        self.ewald = EwaldSum2D(self.supercell_lattice)
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: MoireData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, prev_walker_stats, rngs
        charges = -jnp.ones(
            (data.positions.shape[0],),
            dtype=data.positions.dtype,
        )
        energy = self.ewald.energy(data.positions, charges)
        return {self.output_key: energy * self.coulomb_prefactor_mev}, state


@configurable_dataclass
class MoireSOC(PerWalkerEstimator):
    r"""Estimator for the valley-momentum kinetic-shift correction.

    For Cartesian component :math:`a` and electron :math:`i`, the bilayer drift
    and constant matrices are

    .. math::

        K_i^a = \eta_i\operatorname{diag}(K_+^a, K_-^a),\qquad
        Q = \frac12\operatorname{diag}(|K_+|^2, |K_-|^2),

    where :math:`\eta_i` is the physical-spin sign. Their local contribution is
    :math:`i\sum_{a,i}\nabla_i^a(K_i^a\Psi)/\Psi + \sum_i(Q_i\Psi)/\Psi`.

    Args:
        f_layer_components: Function returning layer orbitals, phase amplitudes,
            and identity orbitals.
        primitive_lattice: Direct primitive-cell lattice vectors.
        nspins: Tuple of ``(n_up, n_down)`` electrons.
        prefactor: Scalar mapping the dimensionless correction to meV.
        output_key: Output statistics key.
    """

    f_layer_components: Callable[
        [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ] = runtime_dep()
    primitive_lattice: jnp.ndarray = runtime_dep()
    nspins: tuple[int, int] = runtime_dep()
    prefactor: float = runtime_dep(default=1.0)
    output_key: str = "energy:soc"

    def evaluate_single_walker(
        self,
        params: Params,
        data: MoireData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        phi, chi, phi0 = self.f_layer_components(params, data)
        if phi.shape[0] != 2:
            raise ValueError("MoireSOC supports exactly two layers.")

        k_plus, k_minus = _moire_valley_momenta(self.primitive_lattice)
        spin_sign = _physical_spin_signs(self.nspins, data.positions.dtype)
        zero = jnp.zeros_like(k_plus)
        k_matrix = (
            spin_sign[None, :, None, None]
            * jnp.stack(
                [
                    jnp.stack([k_plus, zero], axis=-1),
                    jnp.stack([zero, k_minus], axis=-1),
                ],
                axis=-2,
            )[:, None, :, :]
        )
        q_plus = jnp.sum(k_plus**2)
        q_minus = jnp.sum(k_minus**2)
        zero = jnp.zeros_like(q_plus)
        q_matrix = 0.5 * jnp.stack(
            [jnp.stack([q_plus, zero]), jnp.stack([zero, q_minus])]
        )

        def operator_orbitals_fn(
            network_params: Params, walker_data: MoireData
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            layer_phi, layer_chi, layer_phi0 = self.f_layer_components(
                network_params, walker_data
            )
            transformed = jnp.einsum(
                "il,ailm,mdij->adij", layer_chi, k_matrix, layer_phi
            )
            return layer_phi0, transformed

        derivative_ratio = compute_operator_derivative_ratio(
            params, data, operator_orbitals_fn=operator_orbitals_fn
        )
        q_orbitals = jnp.einsum("il,lm,mdij->dij", chi, q_matrix, phi)
        energy = 1j * jnp.einsum("aia->", derivative_ratio) + jnp.sum(
            compute_operator_ratio(phi0, q_orbitals)
        )
        return {self.output_key: energy * self.prefactor}, state


@configurable_dataclass
class MoirePotential(PerWalkerEstimator):
    r"""Estimator for the bilayer moire potential.

    For electron :math:`i`, the layer-basis matrix is

    .. math::

        V_i = \begin{pmatrix}
          \Delta_b & \operatorname{Re}\Delta_T
            + i\eta_i\operatorname{Im}\Delta_T \\
          \operatorname{Re}\Delta_T
            - i\eta_i\operatorname{Im}\Delta_T & \Delta_t
        \end{pmatrix},

    where :math:`\eta_i` is the physical-spin sign. The first-harmonic fields are

    .. math::

        \Delta_{b/t}(\mathbf r)
          = -2V\sum_{i=1,3,5}\cos(\mathbf g_i\cdot\mathbf r \pm \delta),
        \qquad
        \Delta_T(\mathbf r)
          = \omega(1 + e^{i\mathbf g_2\cdot\mathbf r}
                     + e^{i\mathbf g_3\cdot\mathbf r}).

    Args:
        f_layer_components: Function returning layer orbitals, phase amplitudes,
            and identity orbitals.
        primitive_lattice: Direct primitive-cell lattice vectors.
        nspins: Tuple of ``(n_up, n_down)`` electrons.
        output_key: Output statistics key.
        v1_mev: First-harmonic intralayer potential amplitude :math:`V`.
        phi1_rad: First-harmonic phase :math:`\phi_1` in radians.
        omega1_mev: First-harmonic interlayer tunneling amplitude
            :math:`\omega`.
    """

    f_layer_components: Callable[
        [Params, MoireData], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ] = runtime_dep()
    primitive_lattice: jnp.ndarray = runtime_dep()
    nspins: tuple[int, int] = runtime_dep()
    output_key: str = "energy:moire_potential"
    v1_mev: float = runtime_dep(default=0.0)
    phi1_rad: float = runtime_dep(default=0.0)
    omega1_mev: float = runtime_dep(default=0.0)

    def evaluate_single_walker(
        self,
        params: Params,
        data: MoireData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        phi, chi, phi0 = self.f_layer_components(params, data)
        if phi.shape[0] != 2:
            raise ValueError("MoirePotential supports exactly two layers.")

        positions = data.positions
        reci_vec = _moire_first_shell_vectors(self.primitive_lattice)
        g1_proj = positions @ reci_vec[0:5:2].T
        v_bottom = (
            -2.0 * self.v1_mev * jnp.sum(jnp.cos(g1_proj + self.phi1_rad), axis=-1)
        )
        v_top = -2.0 * self.v1_mev * jnp.sum(jnp.cos(g1_proj - self.phi1_rad), axis=-1)
        delta_t = self.omega1_mev * (
            1.0
            + jnp.exp(1j * (positions @ reci_vec[1]))
            + jnp.exp(1j * (positions @ reci_vec[2]))
        )
        spin_sign = _physical_spin_signs(self.nspins, positions.dtype)
        potential_matrix = jnp.stack(
            [
                jnp.stack(
                    [v_bottom, delta_t.real + 1j * spin_sign * delta_t.imag], axis=-1
                ),
                jnp.stack(
                    [delta_t.real - 1j * spin_sign * delta_t.imag, v_top], axis=-1
                ),
            ],
            axis=-2,
        )
        operator_orbitals = jnp.einsum("il,ilm,mdij->dij", chi, potential_matrix, phi)
        energy = jnp.sum(compute_operator_ratio(phi0, operator_orbitals))
        return {self.output_key: energy}, state
