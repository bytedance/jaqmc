# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""Direct NQS Lorentz integral transform for molecular dipole response.

This module follows the correction-vector formulation used in
arXiv:2504.20195: a full neural response wavefunction ``Psi_L`` is optimized so
that ``(H - E0 - omega - i eta) Psi_L`` is parallel to the source
``Phi = (D - <D>) Psi_0``.  Estimators are written for samples drawn from the
source density ``pi_Phi``; the optimizer can then reuse one source pool across
all response energies.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple
from zipfile import BadZipFile

import h5py
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.geometry import obc
from jaqmc.utils.checkpoint import _pytree_key_path, from_npz
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet
from jaqmc.wavefunction.output.orbital import SplitChannelDense

type ResponseApply = Callable[[Params, MoleculeData], jnp.ndarray]
type GroundLogPsi = Callable[[Params, MoleculeData], jnp.ndarray]


class LITStats(NamedTuple):
    """Diagnostics for one source-sampled LIT batch."""

    loss: jnp.ndarray
    fidelity: jnp.ndarray
    reverse_kl: jnp.ndarray
    signed_lit: jnp.ndarray
    broadened: jnp.ndarray
    source_norm: jnp.ndarray
    action_norm: jnp.ndarray
    log_ratio_norm: jnp.ndarray
    correction_overlap: jnp.ndarray
    normalization: jnp.ndarray
    residual_norm: jnp.ndarray
    equation_relative_residual: jnp.ndarray
    ground_energy_mean: jnp.ndarray
    correction_norm: jnp.ndarray
    shifted_hamiltonian_norm: jnp.ndarray
    error_d: jnp.ndarray
    error_d_correction: jnp.ndarray
    error_d_shifted: jnp.ndarray
    error_d_valid: jnp.ndarray
    reweight_ess: jnp.ndarray
    reweight_ess_fraction: jnp.ndarray
    reweight_max_fraction: jnp.ndarray
    invalid_sample_fraction: jnp.ndarray


class WeightedComplexMoments(NamedTuple):
    """Mergeable moments with the mean stored relative to a stable origin.

    Keeping ``origin`` and ``mean_offset`` separate prevents a small weighted
    mean displacement from being rounded away when values share a large
    complex baseline.  ``mean`` remains available as the public reconstructed
    value, while all variance merges operate directly on the small offsets.
    """

    weight_sum: jnp.ndarray
    origin: jnp.ndarray
    mean_offset: jnp.ndarray
    centered_abs2_sum: jnp.ndarray

    @property
    def mean(self) -> jnp.ndarray:
        """Return the reconstructed weighted mean.

        Returns:
            The complex origin plus its small weighted offset.
        """
        return self.origin + self.mean_offset


class LITSourceSums(NamedTuple):
    """Scale-aware source-sampled sums for chunked LIT evaluation.

    The five ratio-moment fields are expressed in units of ``ratio_scale``.
    This keeps the fidelity, reverse-KL, and ESS estimators finite even when
    the unscaled action/source ratio has a very large dynamic range.
    """

    sample_count: jnp.ndarray
    weight_sum: jnp.ndarray
    valid_sample_count: jnp.ndarray
    ratio_scale: jnp.ndarray
    ratio_sum: jnp.ndarray
    ratio_abs2_sum: jnp.ndarray
    psi_weight_sum: jnp.ndarray
    psi_weight_sq_sum: jnp.ndarray
    psi_log_ratio_abs2_sum: jnp.ndarray
    response_conj_over_source_sum: jnp.ndarray
    ground_energy_sum: jnp.ndarray
    response_over_source_moments: WeightedComplexMoments
    hbar_over_source_moments: WeightedComplexMoments
    psi_weight_max: jnp.ndarray


def weighted_complex_moments(
    values: jnp.ndarray,
    weights: jnp.ndarray,
) -> WeightedComplexMoments:
    """Accumulate stable weighted moments without raw second-moment subtraction.

    The first positive-weight value is used as an origin for the weighted mean.
    This preserves small fluctuations around a large common complex value in
    float32.  The second pass then accumulates the centered absolute-square
    moment directly.
    """
    values = jnp.asarray(values)
    weights = jnp.asarray(weights)
    finite = (
        jnp.isfinite(weights)
        & (weights > 0.0)
        & jnp.isfinite(jnp.real(values))
        & jnp.isfinite(jnp.imag(values))
    )
    safe_weights = jnp.where(finite, weights, 0.0)
    weight_sum = jnp.sum(safe_weights)
    has_weight = weight_sum > 0.0
    anchor_index = jnp.argmax(finite)
    origin = jnp.where(
        has_weight,
        values[anchor_index],
        jnp.asarray(0.0, dtype=values.dtype),
    )
    deviations = jnp.where(
        finite,
        values - origin,
        jnp.asarray(0.0, dtype=values.dtype),
    )
    safe_weight_sum = jnp.where(
        has_weight,
        weight_sum,
        jnp.asarray(1.0, dtype=weight_sum.dtype),
    )
    mean_offset = jnp.sum(safe_weights * deviations) / safe_weight_sum
    mean_offset = jnp.where(
        has_weight,
        mean_offset,
        jnp.asarray(0.0, dtype=values.dtype),
    )
    centered_deviations = jnp.where(
        finite,
        deviations - mean_offset,
        jnp.asarray(0.0, dtype=values.dtype),
    )
    centered_abs2_sum = jnp.sum(safe_weights * jnp.abs(centered_deviations) ** 2)
    centered_abs2_sum = jnp.where(
        has_weight,
        centered_abs2_sum,
        jnp.asarray(0.0, dtype=weight_sum.dtype),
    )
    return WeightedComplexMoments(
        weight_sum=weight_sum,
        origin=origin,
        mean_offset=mean_offset,
        centered_abs2_sum=centered_abs2_sum,
    )


def merge_weighted_complex_moments(
    left: WeightedComplexMoments,
    right: WeightedComplexMoments,
) -> WeightedComplexMoments:
    """Merge two weighted complex moment accumulators with Chan's formula."""
    left_weight = left.weight_sum
    right_weight = right.weight_sum
    weight_sum = left_weight + right_weight
    has_weight = weight_sum > 0.0
    safe_weight_sum = jnp.where(
        has_weight,
        weight_sum,
        jnp.asarray(1.0, dtype=weight_sum.dtype),
    )
    zero_mean = jnp.asarray(0.0, dtype=left.origin.dtype)
    origin = jnp.where(
        left_weight > 0.0,
        left.origin,
        jnp.where(
            right_weight > 0.0,
            right.origin,
            zero_mean,
        ),
    )
    left_mean_offset = jnp.where(
        left_weight > 0.0,
        (left.origin - origin) + left.mean_offset,
        zero_mean,
    )
    right_mean_offset = jnp.where(
        right_weight > 0.0,
        (right.origin - origin) + right.mean_offset,
        zero_mean,
    )
    delta = right_mean_offset - left_mean_offset
    mean_offset = left_mean_offset + delta * (right_weight / safe_weight_sum)
    mean_offset = jnp.where(
        has_weight,
        mean_offset,
        zero_mean,
    )
    cross = jnp.abs(delta) ** 2 * left_weight * right_weight / safe_weight_sum
    centered_abs2_sum = left.centered_abs2_sum + right.centered_abs2_sum + cross
    centered_abs2_sum = jnp.where(
        has_weight,
        centered_abs2_sum,
        jnp.asarray(0.0, dtype=weight_sum.dtype),
    )
    return WeightedComplexMoments(
        weight_sum=weight_sum,
        origin=origin,
        mean_offset=mean_offset,
        centered_abs2_sum=centered_abs2_sum,
    )


class LITResponseFermiNet(nn.Module):
    """Complex FermiNet-style response wavefunction ``Psi_L``."""

    nspins: tuple[int, int]
    ndets: int = 16
    hidden_dims_single: tuple[int, ...] = (256, 256, 256, 256)
    hidden_dims_double: tuple[int, ...] = (32, 32, 32, 32)
    use_last_layer: bool = False
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    orbitals_spin_split: bool = True

    def setup(self) -> None:
        self.feature_layer = MoleculeFeatures()
        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        self.backbone_layer = FermiLayers(
            self.nspins,
            hidden_dims,
            use_last_layer=self.use_last_layer,
        )
        self.orbital_layer = _ComplexOrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )
        self.logdet_layer = LogDet()

    def __call__(self, data: MoleculeData) -> jnp.ndarray:
        embedding = self.feature_layer(data.electrons, data.atoms)
        h_one, _ = self.backbone_layer(
            embedding["ae_features"],
            embedding["ee_features"],
        )
        orbitals = self.orbital_layer(h_one)
        orbitals = orbitals * self.envelope_layer(
            embedding["ae_vec"],
            embedding["r_ae"],
        )
        return self.logdet_layer(orbitals)["logpsi"]


class _ComplexOrbitalProjection(nn.Module):
    """Project FermiNet electron features to complex orbital matrices."""

    nspins: tuple[int, int]
    ndets: int
    orbitals_spin_split: bool = True
    use_bias: bool = False

    @nn.compact
    def __call__(self, h_one: jnp.ndarray) -> jnp.ndarray:
        n_electrons = sum(self.nspins)
        features = [self.ndets, n_electrons, 2]
        active_spins = [spin for spin in self.nspins if spin > 0]
        if self.orbitals_spin_split and len(active_spins) > 1:
            orbitals = SplitChannelDense(self.nspins, features, self.use_bias)(h_one)
        else:
            orbitals = nn.DenseGeneral(features, use_bias=self.use_bias)(h_one)
        orbitals = jnp.transpose(orbitals, (1, 0, 2, 3))
        return orbitals[..., 0] + 1j * orbitals[..., 1]


def parity_project_log_amplitude(
    log_psi: jnp.ndarray,
    inverted_log_psi: jnp.ndarray,
    parity: int,
) -> jnp.ndarray:
    r"""Return an exact parity projection of two complex log amplitudes.

    This stably represents

    .. math::

        \log\left[\frac{\psi(X)+p\,\psi(IX)}{2}\right],\qquad p\in\{-1,+1\}.

    A common large log amplitude is factored out before either a complex
    ``log1p`` sum or an ``expm1`` difference is formed.  The latter preserves a
    small odd component when the two unprojected amplitudes nearly cancel.  No
    nonzero amplitude floor is introduced: an exact parity node remains an
    exact zero encoded by a ``-inf`` real log amplitude.

    ``parity`` is a discrete model choice and must be the Python integer ``1``
    or ``-1``.  When this function itself is jitted it must consequently be a
    static argument (or, more commonly, be captured by a closure).
    """
    if parity not in (-1, 1):
        msg = f"parity must be +1 or -1, got {parity!r}."
        raise ValueError(msg)
    log_psi_array = jnp.asarray(log_psi)
    inverted_array = jnp.asarray(inverted_log_psi)
    if log_psi_array.shape != inverted_array.shape:
        msg = (
            "log_psi and inverted_log_psi must have identical shapes, got "
            f"{log_psi_array.shape} and {inverted_array.shape}."
        )
        raise ValueError(msg)
    complex_dtype = jnp.result_type(
        log_psi_array.dtype,
        inverted_array.dtype,
        jnp.complex64,
    )
    log_psi_array = log_psi_array.astype(complex_dtype)
    inverted_array = inverted_array.astype(complex_dtype)
    use_first_as_base = jnp.real(log_psi_array) >= jnp.real(inverted_array)
    base = jnp.where(use_first_as_base, log_psi_array, inverted_array)
    other = jnp.where(use_first_as_base, inverted_array, log_psi_array)
    delta = other - base
    exact_even_node = jnp.zeros_like(jnp.real(delta), dtype=jnp.bool_)
    if parity == 1:
        # ``log1p(exp(delta))`` is stable across a large real dynamic range but
        # still loses the tiny sum when equal-magnitude amplitudes are nearly
        # antiphase.  Around the nearest odd multiple k*pi, instead use
        #
        #   1 + exp(delta) = 1 - exp(delta - i*k*pi)
        #                  = -expm1(delta - i*k*pi).
        #
        # Computing k from phase/pi (rather than a trigonometric phase wrap)
        # makes phases encoded as +/-pi or any explicitly wound odd multiple
        # land on an exact zero in the working dtype.  The regular ``log1p``
        # path is retained in the half-plane around phase zero, where shifting
        # by pi would introduce needless roundoff into an ordinary sum.
        real_dtype = jnp.real(delta).dtype
        pi = jnp.asarray(jnp.pi, dtype=real_dtype)
        phase = jnp.imag(delta)
        phase_in_pi = phase / pi
        nearest_odd_winding = 2.0 * jnp.round((phase_in_pi - 1.0) / 2.0) + 1.0
        nearest_odd_winding = jax.lax.stop_gradient(nearest_odd_winding)
        # Preserve the more accurate direct subtraction for a genuinely nearby
        # phase, while recognizing an explicitly encoded odd winding in units
        # of pi.  XLA may otherwise reassociate ``phase - k*pi`` by a fraction
        # of an ulp and turn an exact parity node into a tiny nonzero amplitude.
        exact_odd_winding = jnp.equal(phase_in_pi, nearest_odd_winding)
        antiphase_residual = jnp.where(
            exact_odd_winding,
            0.0,
            phase - nearest_odd_winding * pi,
        )
        shifted_delta = jnp.real(delta) + 1j * antiphase_residual
        near_antiphase = jnp.abs(antiphase_residual) <= 0.5 * pi
        cancellation_stable_sum = -jnp.expm1(shifted_delta)
        regular_sum = 1.0 + jnp.exp(delta)
        relative_sum = jnp.where(
            near_antiphase,
            cancellation_stable_sum,
            regular_sum,
        )
        relative_log = jnp.log(relative_sum)
        exact_even_node = (
            near_antiphase & jnp.equal(jnp.real(delta), 0.0) & exact_odd_winding
        )
    else:
        # Preserve the original odd-projector expression exactly.  ``expm1``
        # retains relative accuracy when the amplitudes are nearly identical;
        # orientation restores the requested ordering psi(X) - psi(IX) after
        # selecting either one as the numerical base.
        orientation = jnp.where(use_first_as_base, -1.0, 1.0).astype(complex_dtype)
        relative_log = jnp.log(orientation * jnp.expm1(delta))
    projected_log = base + relative_log - jnp.asarray(jnp.log(2.0), dtype=complex_dtype)
    both_encoded_zero = (
        jnp.isneginf(jnp.real(log_psi_array))
        & jnp.isfinite(jnp.imag(log_psi_array))
        & jnp.isneginf(jnp.real(inverted_array))
        & jnp.isfinite(jnp.imag(inverted_array))
    )
    encoded_zero = jnp.asarray(-jnp.inf + 0.0j, dtype=complex_dtype)
    return jnp.where(
        both_encoded_zero | exact_even_node,
        encoded_zero,
        projected_log,
    )


def odd_parity_project_log_amplitude(
    log_psi: jnp.ndarray,
    inverted_log_psi: jnp.ndarray,
) -> jnp.ndarray:
    r"""Return ``log[(psi(X) - psi(IX)) / 2]`` stably.

    This compatibility wrapper is equivalent to
    :func:`parity_project_log_amplitude` with ``parity=-1``.
    """
    return parity_project_log_amplitude(log_psi, inverted_log_psi, -1)


def parity_log_amplitude_residual(
    log_psi: jnp.ndarray,
    inverted_log_psi: jnp.ndarray,
    parity: int,
    *,
    epsilon: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Return a scale-invariant parity residual for every batch element.

    For scalar log amplitudes of any matching shape, the returned array has the
    same shape and contains

    .. math::

        r_p(X) =
        \frac{|\psi(IX)-p\,\psi(X)|^2}
        {|\psi(IX)|^2+|\psi(X)|^2}.

    A separate common log scale is removed from every pair before
    exponentiation.  Thus the diagnostic is invariant under arbitrary common
    complex rescaling, remains finite across the float32 exponential range,
    and assigns zero residual when both amplitudes are encoded zeros.  Invalid
    log amplitudes propagate as ``nan`` for the affected sample.  An exact
    parity eigenstate has residual zero, while the opposite parity has residual
    two.
    """
    if parity not in (-1, 1):
        msg = f"parity must be +1 or -1, got {parity!r}."
        raise ValueError(msg)
    log_psi_array = jnp.asarray(log_psi)
    inverted_array = jnp.asarray(inverted_log_psi)
    if log_psi_array.shape != inverted_array.shape:
        msg = (
            "log_psi and inverted_log_psi must have identical shapes, got "
            f"{log_psi_array.shape} and {inverted_array.shape}."
        )
        raise ValueError(msg)

    complex_dtype = jnp.result_type(
        log_psi_array.dtype,
        inverted_array.dtype,
        jnp.complex64,
    )
    paired_logs = jnp.stack(
        (
            log_psi_array.astype(complex_dtype),
            inverted_array.astype(complex_dtype),
        ),
        axis=-1,
    )
    real_logs = jnp.real(paired_logs)
    imag_logs = jnp.imag(paired_logs)
    finite = jnp.isfinite(real_logs) & jnp.isfinite(imag_logs)
    encoded_zero = jnp.isneginf(real_logs) & jnp.isfinite(imag_logs)
    invalid = ~(finite | encoded_zero)
    masked_real = jnp.where(finite, real_logs, -jnp.inf)
    log_scale = jnp.max(masked_real, axis=-1, keepdims=True)
    log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
    log_scale = jax.lax.stop_gradient(log_scale)
    safe_delta = jnp.where(finite, paired_logs - log_scale, 0.0 + 0.0j)
    amplitudes = jnp.where(finite, jnp.exp(safe_delta), 0.0 + 0.0j)
    psi = amplitudes[..., 0]
    psi_at_inversion = amplitudes[..., 1]

    numerator = jnp.abs(psi_at_inversion - parity * psi) ** 2
    denominator = jnp.abs(psi_at_inversion) ** 2 + jnp.abs(psi) ** 2
    if epsilon is None:
        epsilon_array = jnp.asarray(
            16.0 * jnp.finfo(jnp.real(psi).dtype).eps,
            dtype=denominator.dtype,
        )
    else:
        epsilon_array = jnp.asarray(epsilon, dtype=denominator.dtype)
    residual = numerator / jnp.maximum(denominator, epsilon_array)
    invalid_sample = jnp.any(invalid, axis=-1)
    return jnp.where(
        invalid_sample,
        jnp.asarray(jnp.nan, dtype=residual.dtype),
        residual,
    )


def parity_log_amplitude_loss(
    log_psi: jnp.ndarray,
    inverted_log_psi: jnp.ndarray,
    parity: int,
    *,
    epsilon: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the mean scale-invariant parity residual over a batch."""
    return jnp.mean(
        parity_log_amplitude_residual(
            log_psi,
            inverted_log_psi,
            parity,
            epsilon=epsilon,
        )
    )


def molecular_electronic_dipole(data: MoleculeData, axis: int) -> jnp.ndarray:
    """Electronic dipole component for a fixed-nuclei transition source."""
    return -jnp.sum(data.electrons[:, int(axis)])


def molecular_potential_energy(data: MoleculeData) -> jnp.ndarray:
    """Molecular Coulomb potential energy for one walker."""
    nelec = data.electrons.shape[0]
    natom = data.atoms.shape[0]
    r_ae = obc.pair_displacements_between(data.electrons, data.atoms)[1]
    r_ee = obc.pair_displacements_within(data.electrons)[1] + jnp.eye(nelec)
    r_aa = obc.pair_displacements_within(data.atoms)[1] + jnp.eye(natom)
    return (
        jnp.sum(-jnp.ones(nelec)[:, None] * data.charges / r_ae)
        + jnp.sum(jnp.triu(1 / r_ee, k=1))
        + jnp.sum(jnp.triu(data.charges * data.charges[:, None] / r_aa, k=1))
    )


def _flat_electrons(data: MoleculeData) -> tuple[jnp.ndarray, tuple[int, ...]]:
    shape = data.electrons.shape
    return jnp.ravel(data.electrons), shape


def _with_flat_electrons(
    data: MoleculeData, flat_electrons: jnp.ndarray, shape: tuple[int, ...]
) -> MoleculeData:
    return data.merge({"electrons": jnp.reshape(flat_electrons, shape)})


def ground_local_energy(
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    data: MoleculeData,
) -> jnp.ndarray:
    """Evaluate the ground-state local energy by coordinate derivatives."""
    flat, shape = _flat_electrons(data)

    def logpsi_flat(x):
        value = ground_logpsi(ground_params, _with_flat_electrons(data, x, shape))
        return jnp.real(value)

    grad_log = jax.grad(logpsi_flat)(flat)
    hess_log = jax.hessian(logpsi_flat)(flat)
    kinetic = -0.5 * (jnp.trace(hess_log) + jnp.dot(grad_log, grad_log))
    return kinetic + molecular_potential_energy(data)


def local_action_ratio(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    data: MoleculeData,
    *,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``((H-z) Psi_L / Psi_0, Psi_L / Psi_0, E_L[Psi_L])``."""
    flat, shape = _flat_electrons(data)

    def ground_log_flat(x):
        return ground_logpsi(ground_params, _with_flat_electrons(data, x, shape))

    def response_log_flat(x):
        return response_apply(response_params, _with_flat_electrons(data, x, shape))

    response_log = response_log_flat(flat)
    log_ratio = response_log - ground_log_flat(flat)
    response_ratio = jnp.exp(log_ratio)
    grad_re = jax.grad(lambda x: jnp.real(response_log_flat(x)))(flat)
    grad_im = jax.grad(lambda x: jnp.imag(response_log_flat(x)))(flat)
    grad_log_response = grad_re + 1j * grad_im
    hess_re = jax.hessian(lambda x: jnp.real(response_log_flat(x)))(flat)
    hess_im = jax.hessian(lambda x: jnp.imag(response_log_flat(x)))(flat)
    lap_log_response = jnp.trace(hess_re) + 1j * jnp.trace(hess_im)
    local_energy = -0.5 * (
        lap_log_response + jnp.dot(grad_log_response, grad_log_response)
    ) + molecular_potential_energy(data)
    shift = jnp.asarray(omega, dtype=response_ratio.real.dtype) + 1j * jnp.asarray(
        eta,
        dtype=response_ratio.real.dtype,
    )
    action = response_ratio * (
        local_energy - jnp.asarray(ground_energy, dtype=response_ratio.dtype) - shift
    )
    return action, response_ratio, local_energy


def source_sampled_stats(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    batched_data: BatchedData[MoleculeData],
    *,
    axis: int,
    source_center: float | jnp.ndarray,
    source_norm: float | jnp.ndarray,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    source_floor: float | jnp.ndarray = 0.0,
    eps: float = 1e-12,
) -> LITStats:
    """Compute fidelity and LIT observables from ``pi_Phi`` samples."""
    sums = source_sampled_sums(
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        axis=axis,
        source_center=source_center,
        ground_energy=ground_energy,
        omega=omega,
        eta=eta,
        source_floor=source_floor,
        eps=eps,
    )
    return stats_from_source_sums(
        sums,
        source_norm=source_norm,
        omega=omega,
        eta=eta,
        eps=eps,
    )


def source_sampled_sums(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    batched_data: BatchedData[MoleculeData],
    *,
    axis: int,
    source_center: float | jnp.ndarray,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    source_floor: float | jnp.ndarray = 0.0,
    eps: float = 1e-12,
) -> LITSourceSums:
    """Return additive raw sums for source-sampled LIT observables."""
    data = batched_data.data
    action, response_ratio, eloc_response = jax.vmap(
        lambda one: local_action_ratio(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            one,
            ground_energy=ground_energy,
            omega=omega,
            eta=eta,
        ),
        in_axes=(batched_data.vmap_axis,),
    )(data)
    dipole = jax.vmap(
        lambda one: molecular_electronic_dipole(one, axis),
        in_axes=(batched_data.vmap_axis,),
    )(data)
    source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
    floor = jnp.asarray(source_floor, dtype=dipole.dtype)
    sampled_source_abs = jnp.maximum(jnp.abs(source), floor)
    source_weight = (jnp.abs(source) / jnp.maximum(sampled_source_abs, eps)) ** 2
    base_finite = (
        jnp.isfinite(jnp.real(action))
        & jnp.isfinite(jnp.imag(action))
        & jnp.isfinite(jnp.real(response_ratio))
        & jnp.isfinite(jnp.imag(response_ratio))
        & jnp.isfinite(source)
        & jnp.isfinite(source_weight)
    )
    safe_source = jnp.where(
        jnp.abs(source) > eps,
        source,
        jnp.asarray(eps, dtype=source.dtype) * jnp.where(source < 0, -1.0, 1.0),
    )
    ratio = action / safe_source
    finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
    finite = base_finite & finite_ratio
    action = jnp.where(finite, action, jnp.asarray(0.0, dtype=action.dtype))
    response_ratio = jnp.where(
        finite,
        response_ratio,
        jnp.asarray(0.0, dtype=response_ratio.dtype),
    )
    source_weight = jnp.where(
        finite,
        source_weight,
        jnp.asarray(0.0, dtype=source_weight.dtype),
    )
    ratio = jnp.where(finite, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
    ratio_abs = jnp.where(source_weight > 0.0, jnp.abs(ratio), 0.0)
    max_ratio_abs = jnp.max(ratio_abs)
    ratio_scale = jnp.where(
        max_ratio_abs > 0.0,
        max_ratio_abs,
        jnp.asarray(1.0, dtype=ratio_abs.dtype),
    )
    scaled_ratio = ratio / jax.lax.stop_gradient(ratio_scale)
    scaled_ratio_abs2 = jnp.abs(scaled_ratio) ** 2
    psi_weight_unnormalized = source_weight * scaled_ratio_abs2
    log_ratio_abs2 = 2.0 * jnp.log(
        jnp.maximum(jnp.abs(scaled_ratio), jnp.asarray(eps, dtype=ratio_abs.dtype))
    )
    shift = jnp.asarray(omega, dtype=response_ratio.real.dtype) + 1j * jnp.asarray(
        eta,
        dtype=response_ratio.real.dtype,
    )
    hbar_response_ratio = action + shift * response_ratio
    response_over_source = response_ratio / safe_source
    hbar_over_source = hbar_response_ratio / safe_source
    response_over_source_moments = weighted_complex_moments(
        response_over_source,
        source_weight,
    )
    hbar_over_source_moments = weighted_complex_moments(
        hbar_over_source,
        source_weight,
    )
    eloc_finite = jnp.isfinite(jnp.real(eloc_response))
    eloc_response = jnp.where(
        eloc_finite,
        eloc_response,
        jnp.asarray(0.0, dtype=eloc_response.dtype),
    )
    sample_count = jnp.asarray(action.shape[0], dtype=source_weight.real.dtype)
    return LITSourceSums(
        sample_count=sample_count,
        weight_sum=jnp.sum(source_weight),
        valid_sample_count=jnp.sum(finite),
        ratio_scale=ratio_scale,
        ratio_sum=jnp.sum(source_weight * scaled_ratio),
        ratio_abs2_sum=jnp.sum(source_weight * scaled_ratio_abs2),
        psi_weight_sum=jnp.sum(psi_weight_unnormalized),
        psi_weight_sq_sum=jnp.sum(psi_weight_unnormalized**2),
        psi_log_ratio_abs2_sum=jnp.sum(psi_weight_unnormalized * log_ratio_abs2),
        response_conj_over_source_sum=jnp.sum(
            source_weight * jnp.conj(response_ratio) / safe_source
        ),
        ground_energy_sum=jnp.real(jnp.sum(eloc_response)),
        response_over_source_moments=response_over_source_moments,
        hbar_over_source_moments=hbar_over_source_moments,
        psi_weight_max=jnp.max(psi_weight_unnormalized),
    )


def merge_source_sums(
    left: LITSourceSums,
    right: LITSourceSums,
) -> LITSourceSums:
    """Merge source sums, including scaled ratios and centered moments."""
    scale = jnp.maximum(left.ratio_scale, right.ratio_scale)
    tiny = jnp.asarray(jnp.finfo(scale.dtype).tiny, dtype=scale.dtype)
    scale = jnp.maximum(scale, tiny)
    left_factor = left.ratio_scale / scale
    right_factor = right.ratio_scale / scale

    def log_moment(moment, weight, factor):
        factor_sq = factor**2
        safe_factor = jnp.maximum(factor, tiny)
        return factor_sq * (moment + 2.0 * jnp.log(safe_factor) * weight)

    left_additive = left._replace(
        response_over_source_moments=jax.tree.map(
            jnp.zeros_like,
            left.response_over_source_moments,
        ),
        hbar_over_source_moments=jax.tree.map(
            jnp.zeros_like,
            left.hbar_over_source_moments,
        ),
        psi_weight_max=jnp.zeros_like(left.psi_weight_max),
    )
    right_additive = right._replace(
        response_over_source_moments=jax.tree.map(
            jnp.zeros_like,
            right.response_over_source_moments,
        ),
        hbar_over_source_moments=jax.tree.map(
            jnp.zeros_like,
            right.hbar_over_source_moments,
        ),
        psi_weight_max=jnp.zeros_like(right.psi_weight_max),
    )
    merged = jax.tree.map(operator.add, left_additive, right_additive)

    return merged._replace(
        ratio_scale=scale,
        ratio_sum=left_factor * left.ratio_sum + right_factor * right.ratio_sum,
        ratio_abs2_sum=(
            left_factor**2 * left.ratio_abs2_sum
            + right_factor**2 * right.ratio_abs2_sum
        ),
        psi_weight_sum=(
            left_factor**2 * left.psi_weight_sum
            + right_factor**2 * right.psi_weight_sum
        ),
        psi_weight_sq_sum=(
            left_factor**4 * left.psi_weight_sq_sum
            + right_factor**4 * right.psi_weight_sq_sum
        ),
        psi_log_ratio_abs2_sum=(
            log_moment(
                left.psi_log_ratio_abs2_sum,
                left.psi_weight_sum,
                left_factor,
            )
            + log_moment(
                right.psi_log_ratio_abs2_sum,
                right.psi_weight_sum,
                right_factor,
            )
        ),
        response_over_source_moments=merge_weighted_complex_moments(
            left.response_over_source_moments,
            right.response_over_source_moments,
        ),
        hbar_over_source_moments=merge_weighted_complex_moments(
            left.hbar_over_source_moments,
            right.hbar_over_source_moments,
        ),
        psi_weight_max=jnp.maximum(
            left_factor**2 * left.psi_weight_max,
            right_factor**2 * right.psi_weight_max,
        ),
    )


def _merge_weighted_complex_moments_across_devices(
    moments: WeightedComplexMoments,
    *,
    axis_name: str,
) -> WeightedComplexMoments:
    """Merge centered moments with a VMA-safe parallel Chan reduction.

    A common origin is selected from the first nonempty shard.  Only deviations
    from that origin are collectively summed, which retains the numerical
    stability of Chan's merge for nearly collinear, large-magnitude means.  The
    implementation uses only scalar collectives so it also works inside
    ``shard_map`` with varying-manual-axis checking enabled.
    """
    axis_index = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(
        jnp.asarray(1, dtype=axis_index.dtype),
        axis_name=axis_name,
    )
    has_weight = moments.weight_sum > 0.0
    first_nonempty = jax.lax.pmin(
        jnp.where(has_weight, axis_index, axis_size),
        axis_name=axis_name,
    )
    zero_mean = jnp.asarray(0.0, dtype=moments.origin.dtype)
    origin = jax.lax.psum(
        jnp.where(axis_index == first_nonempty, moments.origin, zero_mean),
        axis_name=axis_name,
    )
    weight_sum = jax.lax.psum(moments.weight_sum, axis_name=axis_name)
    safe_weight_sum = jnp.where(
        weight_sum > 0.0,
        weight_sum,
        jnp.asarray(1.0, dtype=weight_sum.dtype),
    )
    local_delta = jnp.where(
        has_weight,
        (moments.origin - origin) + moments.mean_offset,
        zero_mean,
    )
    mean_delta = (
        jax.lax.psum(
            moments.weight_sum * local_delta,
            axis_name=axis_name,
        )
        / safe_weight_sum
    )
    between_shard = jnp.where(
        has_weight,
        moments.weight_sum * jnp.abs(local_delta - mean_delta) ** 2,
        jnp.asarray(0.0, dtype=moments.centered_abs2_sum.dtype),
    )
    centered_abs2_sum = jax.lax.psum(
        moments.centered_abs2_sum + between_shard,
        axis_name=axis_name,
    )
    return WeightedComplexMoments(
        weight_sum=weight_sum,
        origin=origin,
        mean_offset=mean_delta,
        centered_abs2_sum=centered_abs2_sum,
    )


def merge_source_sums_across_devices(
    local_sums: LITSourceSums,
    *,
    axis_name: str,
) -> LITSourceSums:
    """Collectively merge source sums over a named data-parallel axis."""
    local_has_ratio_mass = (
        (local_sums.ratio_abs2_sum > 0.0)
        | (local_sums.psi_weight_sum > 0.0)
        | (jnp.abs(local_sums.ratio_sum) > 0.0)
    )
    local_scale = jnp.where(
        local_has_ratio_mass,
        local_sums.ratio_scale,
        jnp.asarray(0.0, dtype=local_sums.ratio_scale.dtype),
    )
    scale = jax.lax.pmax(local_scale, axis_name=axis_name)
    scale = jnp.where(
        scale > 0.0,
        scale,
        jnp.asarray(1.0, dtype=scale.dtype),
    )
    tiny = jnp.asarray(jnp.finfo(scale.dtype).tiny, dtype=scale.dtype)
    factor = local_sums.ratio_scale / jnp.maximum(scale, tiny)
    factor = jnp.where(local_has_ratio_mass, factor, 0.0)
    factor_sq = factor**2
    safe_factor = jnp.maximum(factor, tiny)

    additive = local_sums._replace(
        response_over_source_moments=jax.tree.map(
            jnp.zeros_like,
            local_sums.response_over_source_moments,
        ),
        hbar_over_source_moments=jax.tree.map(
            jnp.zeros_like,
            local_sums.hbar_over_source_moments,
        ),
        psi_weight_max=jnp.zeros_like(local_sums.psi_weight_max),
    )
    merged = jax.tree.map(
        lambda value: jax.lax.psum(value, axis_name=axis_name),
        additive,
    )
    return merged._replace(
        ratio_scale=scale,
        ratio_sum=jax.lax.psum(
            factor * local_sums.ratio_sum,
            axis_name=axis_name,
        ),
        ratio_abs2_sum=jax.lax.psum(
            factor_sq * local_sums.ratio_abs2_sum,
            axis_name=axis_name,
        ),
        psi_weight_sum=jax.lax.psum(
            factor_sq * local_sums.psi_weight_sum,
            axis_name=axis_name,
        ),
        psi_weight_sq_sum=jax.lax.psum(
            factor_sq**2 * local_sums.psi_weight_sq_sum,
            axis_name=axis_name,
        ),
        psi_log_ratio_abs2_sum=jax.lax.psum(
            factor_sq
            * (
                local_sums.psi_log_ratio_abs2_sum
                + 2.0 * jnp.log(safe_factor) * local_sums.psi_weight_sum
            ),
            axis_name=axis_name,
        ),
        response_over_source_moments=(
            _merge_weighted_complex_moments_across_devices(
                local_sums.response_over_source_moments,
                axis_name=axis_name,
            )
        ),
        hbar_over_source_moments=(
            _merge_weighted_complex_moments_across_devices(
                local_sums.hbar_over_source_moments,
                axis_name=axis_name,
            )
        ),
        psi_weight_max=jax.lax.pmax(
            factor_sq * local_sums.psi_weight_max,
            axis_name=axis_name,
        ),
    )


def stats_from_source_sums(
    sums: LITSourceSums,
    *,
    source_norm: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    eps: float = 1e-12,
) -> LITStats:
    """Convert additive source-sampled sums into standard diagnostics."""
    real_dtype = sums.weight_sum.dtype
    safe_weight_sum = jnp.maximum(
        sums.weight_sum,
        jnp.asarray(eps, dtype=real_dtype),
    )
    ratio_scale = jnp.maximum(
        sums.ratio_scale,
        jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
    )
    scaled_normalization = sums.ratio_sum / safe_weight_sum
    scaled_ratio_norm = sums.ratio_abs2_sum / safe_weight_sum
    has_action_mass = (
        jnp.isfinite(scaled_ratio_norm)
        & jnp.isfinite(ratio_scale)
        & (sums.psi_weight_sum > jnp.asarray(0.0, dtype=real_dtype))
    )
    normalization = ratio_scale * scaled_normalization
    ratio_norm = ratio_scale**2 * scaled_ratio_norm
    log_ratio_norm = 2.0 * jnp.log(ratio_scale) + jnp.log(
        jnp.maximum(
            scaled_ratio_norm,
            jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
        )
    )
    reweight_ess = sums.psi_weight_sum**2 / jnp.maximum(
        sums.psi_weight_sq_sum,
        jnp.asarray(eps, dtype=real_dtype),
    )
    valid_sample_count = jnp.maximum(
        sums.valid_sample_count,
        jnp.asarray(1, dtype=real_dtype),
    )
    reweight_ess_fraction = reweight_ess / valid_sample_count
    reweight_max_fraction = sums.psi_weight_max / jnp.where(
        sums.psi_weight_sum > 0.0,
        sums.psi_weight_sum,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    reweight_max_fraction = jnp.where(
        has_action_mass & jnp.isfinite(reweight_max_fraction),
        reweight_max_fraction,
        jnp.asarray(jnp.nan, dtype=real_dtype),
    )
    fidelity = (jnp.abs(scaled_normalization) ** 2) / jnp.maximum(
        scaled_ratio_norm,
        jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
    )
    fidelity = jnp.clip(jnp.real(fidelity), 0.0, 1.0)
    reverse_kl = sums.psi_log_ratio_abs2_sum / jnp.maximum(
        sums.psi_weight_sum, jnp.asarray(eps, dtype=real_dtype)
    ) - jnp.log(
        jnp.maximum(
            scaled_ratio_norm,
            jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
        )
    )
    reverse_kl = jnp.where(
        has_action_mass,
        jnp.maximum(
            jnp.real(reverse_kl),
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    loss = 1.0 - fidelity

    phi_norm = jnp.asarray(source_norm, dtype=real_dtype)
    action_norm = phi_norm * ratio_norm
    correction_overlap = phi_norm * sums.response_conj_over_source_sum / safe_weight_sum
    normalization_valid = (
        has_action_mass
        & jnp.isfinite(jnp.real(normalization))
        & jnp.isfinite(jnp.imag(normalization))
        & (jnp.abs(normalization) > 0.0)
        & jnp.isfinite(jnp.real(correction_overlap))
        & jnp.isfinite(jnp.imag(correction_overlap))
        & jnp.isfinite(jnp.asarray(eta, dtype=real_dtype))
        & (jnp.asarray(eta, dtype=real_dtype) > 0.0)
    )
    safe_normalization = jnp.where(
        normalization_valid,
        normalization,
        jnp.asarray(1.0 + 0.0j, dtype=normalization.dtype),
    )
    normalized_overlap = correction_overlap / jnp.conj(safe_normalization)
    signed_lit = jnp.where(
        normalization_valid,
        -jnp.imag(normalized_overlap) / jnp.asarray(eta),
        jnp.asarray(jnp.nan, dtype=real_dtype),
    )
    broadened = jnp.asarray(eta) * signed_lit / jnp.pi
    scaled_normalization_abs2 = jnp.abs(scaled_normalization) ** 2
    residual_valid = (
        has_action_mass
        & jnp.isfinite(scaled_normalization_abs2)
        & (scaled_normalization_abs2 > 0.0)
    )
    safe_scaled_normalization_abs2 = jnp.where(
        residual_valid,
        scaled_normalization_abs2,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    residual_mean = jnp.where(
        residual_valid,
        jnp.maximum(
            jnp.real(scaled_ratio_norm / safe_scaled_normalization_abs2 - 1.0),
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        jnp.asarray(jnp.nan, dtype=real_dtype),
    )
    residual_norm = phi_norm * residual_mean
    equation_relative_residual = jnp.sqrt(residual_mean)
    (
        correction_norm,
        shifted_hamiltonian_norm,
        error_d_correction,
        error_d_shifted,
        error_d,
        error_d_valid,
    ) = _source_sampled_error_d_sums(
        sums,
        phi_norm,
        omega=omega,
        eta=eta,
    )
    safe_sample_count = jnp.maximum(
        sums.sample_count,
        jnp.asarray(1, dtype=real_dtype),
    )
    invalid_sample_fraction = jnp.maximum(
        1.0 - sums.valid_sample_count / safe_sample_count,
        jnp.asarray(0.0, dtype=real_dtype),
    )
    invalid_sample_fraction = jnp.where(
        has_action_mass,
        invalid_sample_fraction,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    return LITStats(
        loss=jnp.real(loss),
        fidelity=fidelity,
        reverse_kl=reverse_kl,
        signed_lit=jnp.real(signed_lit),
        broadened=jnp.real(broadened),
        source_norm=jnp.real(phi_norm),
        action_norm=jnp.real(action_norm),
        log_ratio_norm=jnp.real(log_ratio_norm),
        correction_overlap=correction_overlap,
        normalization=normalization,
        residual_norm=jnp.real(residual_norm),
        equation_relative_residual=jnp.real(equation_relative_residual),
        ground_energy_mean=jnp.real(sums.ground_energy_sum / safe_sample_count),
        correction_norm=jnp.real(correction_norm),
        shifted_hamiltonian_norm=jnp.real(shifted_hamiltonian_norm),
        error_d=jnp.real(error_d),
        error_d_correction=jnp.real(error_d_correction),
        error_d_shifted=jnp.real(error_d_shifted),
        error_d_valid=error_d_valid,
        reweight_ess=jnp.real(reweight_ess),
        reweight_ess_fraction=jnp.real(reweight_ess_fraction),
        reweight_max_fraction=jnp.real(reweight_max_fraction),
        invalid_sample_fraction=jnp.real(invalid_sample_fraction),
    )


def _source_sampled_error_d_sums(
    sums: LITSourceSums,
    phi_norm: jnp.ndarray,
    *,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Evaluate the two paper error factors from centered source moments.

    Raw ``E[|X|^2] - |E[X]|^2`` subtraction is deliberately avoided.  It loses
    all fluctuation information for nearly collinear response/source ratios in
    float32.  These are the raw norms in Eq. (19) of the supplement: the action
    normalization ``N`` is applied once, later, by the LIT error monitor.
    """
    response_moments = sums.response_over_source_moments
    hbar_moments = sums.hbar_over_source_moments
    real_dtype = phi_norm.dtype
    nan = jnp.asarray(jnp.nan, dtype=real_dtype)

    shift_abs2 = (
        jnp.asarray(omega, dtype=real_dtype) ** 2
        + jnp.asarray(eta, dtype=real_dtype) ** 2
    )
    response_weight = response_moments.weight_sum
    hbar_weight = hbar_moments.weight_sum
    response_m2 = response_moments.centered_abs2_sum
    hbar_m2 = hbar_moments.centered_abs2_sum
    valid = (
        jnp.isfinite(phi_norm)
        & (phi_norm > 0.0)
        & jnp.isfinite(shift_abs2)
        & (shift_abs2 > 0.0)
        & jnp.isfinite(response_weight)
        & (response_weight > 0.0)
        & jnp.isfinite(hbar_weight)
        & (hbar_weight > 0.0)
        & jnp.isfinite(jnp.real(response_moments.mean))
        & jnp.isfinite(jnp.imag(response_moments.mean))
        & jnp.isfinite(jnp.real(hbar_moments.mean))
        & jnp.isfinite(jnp.imag(hbar_moments.mean))
        & jnp.isfinite(response_m2)
        & (response_m2 >= 0.0)
        & jnp.isfinite(hbar_m2)
        & (hbar_m2 >= 0.0)
    )

    # Denominators are made benign only for evaluating the masked invalid
    # branch.  They are never used as numerical regularizers for valid data.
    safe_shift_abs2 = jnp.where(valid, shift_abs2, 1.0)
    safe_response_weight = jnp.where(valid, response_weight, 1.0)
    safe_hbar_weight = jnp.where(valid, hbar_weight, 1.0)

    correction_variance = phi_norm * response_m2 / safe_response_weight
    shifted_variance = phi_norm * hbar_m2 / safe_hbar_weight / safe_shift_abs2
    correction_norm = phi_norm * (
        response_m2 / safe_response_weight + jnp.abs(response_moments.mean) ** 2
    )
    shifted_hamiltonian_norm = (
        phi_norm
        * (hbar_m2 / safe_hbar_weight + jnp.abs(hbar_moments.mean) ** 2)
        / safe_shift_abs2
    )
    error_d_correction = jnp.sqrt(correction_variance)
    error_d_shifted = jnp.sqrt(shifted_variance)
    error_d = jnp.minimum(error_d_correction, error_d_shifted)

    return (
        jnp.where(valid, correction_norm, nan),
        jnp.where(valid, shifted_hamiltonian_norm, nan),
        jnp.where(valid, error_d_correction, nan),
        jnp.where(valid, error_d_shifted, nan),
        jnp.where(valid, error_d, nan),
        valid,
    )


def restore_params_from_checkpoint(
    checkpoint_path: str | Path | UPath,
    fallback_params: Params,
    *,
    state_field: str = "params",
) -> tuple[int, Params]:
    """Restore a parameter subtree from a JaQMC stage checkpoint."""
    path = UPath(checkpoint_path)
    if path.is_dir():
        ckpt_files = sorted(path.glob("*ckpt_*.npz"), reverse=True)
        if not ckpt_files:
            msg = f"No checkpoint files found in {path}"
            raise FileNotFoundError(msg)
        path = ckpt_files[0]
    if not path.is_file():
        msg = f"Checkpoint path does not exist: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as f:
        try:
            with np.load(f) as npf:
                step = int(npf["step"].item()) if "step" in npf else -1
                params = _restore_prefixed_tree(npf, fallback_params, state_field)
        except (OSError, EOFError, BadZipFile) as exc:
            msg = f"Failed to restore checkpoint {path}"
            raise ValueError(msg) from exc
    return step, params


def _restore_prefixed_tree(
    npf: Mapping[str, np.ndarray | h5py.Group],
    fallback: Params,
    state_field: str,
) -> Params:
    ref_vals_with_path, treedef = jax.tree_util.tree_flatten_with_path(fallback)
    restored = []
    for key_path, ref_val in ref_vals_with_path:
        name = _pytree_key_path(key_path)
        candidates = (f"{state_field}/{name}", name)
        for candidate in candidates:
            if candidate in npf:
                restored.append(from_npz(candidate, npf, ref_val))
                break
        else:
            msg = (
                f"Checkpoint is missing parameter leaf {candidates[0]!r} "
                f"(or fallback key {name!r})"
            )
            raise KeyError(msg)
    return jax.tree.unflatten(treedef, restored)
