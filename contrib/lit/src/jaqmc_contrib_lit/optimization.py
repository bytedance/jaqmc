# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""SPRING, SR, and held-out selection numerics for LIT."""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping

import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from jax.flatten_util import ravel_pytree

from jaqmc.utils import parallel_jax
from jaqmc_contrib_lit.common import _AXIS_NAMES
from jaqmc_contrib_lit.state import (
    _SPRING_PARAMETER_GROUPS,
    _SourceDistillationStats,
    _SpringOptimizerDiagnostics,
    _SpringState,
)
from jaqmc_contrib_lit.transform import lit_error_bound

try:
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64  # type: ignore[no-redef]

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


def _log_spring_optimizer_diagnostics(
    diagnostics: _SpringOptimizerDiagnostics | None,
    *,
    axis: int,
    stage: str,
    omega: float,
    iteration: int,
) -> None:
    """Log the most recent source-sampled SPRING scalar diagnostics."""
    if diagnostics is None:
        return
    host = jax.device_get(diagnostics)
    if not bool(host.available):
        return
    logger.info(
        "axis=%s stage=%s omega=%.6f iter=%d spring_grad=%.6e "
        "spring_grad_fidelity=%.6e spring_grad_kl_weighted=%.6e "
        "spring_fidelity_kl_cosine=%.6f spring_gradient_cancellation=%.6e "
        "spring_direction=%.6e spring_update=%.6e spring_clip_factor=%.6e "
        "spring_clipped=%d spring_damping=%.6e spring_qfi_mean_diagonal=%.6e "
        "spring_history_gradient_ratio=%.6e response_grad_rms=%.6e "
        "response_update=%.6e",
        _AXIS_NAMES[axis],
        stage,
        omega,
        iteration,
        float(host.combined_gradient_norm),
        float(host.fidelity_gradient_norm),
        float(host.weighted_reverse_kl_gradient_norm),
        float(host.fidelity_kl_cosine),
        float(host.gradient_cancellation_ratio),
        float(host.direction_norm),
        float(host.update_norm),
        float(host.clip_factor),
        int(host.clip_factor < 1.0),
        float(host.damping),
        float(host.mean_metric_diagonal),
        float(host.history_gradient_ratio),
        float(host.parameter_group_gradient_rms[0]),
        float(host.parameter_group_update_norm[0]),
    )


def _solve_sr_direction_data_parallel(
    local_score_aug,
    grad_flat,
    damping,
    *,
    device_count: int,
    local_kernel_null_vectors=None,
    kernel_projector_scale=1.0,
    axis_name: str = parallel_jax.BATCH_AXIS_NAME,
):
    """Solve the exact global SR system without gathering the full score.

    The dual branch redistributes parameter columns with ``all_to_all``.  Each
    device therefore owns all global sample rows for only a fraction of the
    parameters, constructs one Gram contribution, and replicates only the
    ``O(batch**2)`` kernel and Cholesky factor.

    Returns:
        The replicated flattened SR direction.

    Raises:
        ValueError: If ``device_count`` is not positive.
    """
    if device_count < 1:
        msg = "device_count must be positive."
        raise ValueError(msg)
    parameter_count = int(grad_flat.shape[0])
    local_sample_count = int(local_score_aug.shape[0])
    sample_count = local_sample_count * int(device_count)
    original_dtype = grad_flat.dtype
    with _enable_x64(True):
        solve_dtype = jnp.float64
        score_solve = local_score_aug.astype(solve_dtype)
        grad_solve = grad_flat.astype(solve_dtype)
        damping_solve = jnp.asarray(damping, dtype=solve_dtype)
        if parameter_count <= sample_count:
            metric = jax.lax.psum(
                score_solve.T @ score_solve,
                axis_name=axis_name,
            )
            metric = (metric + metric.T) / 2.0
            metric = metric + damping_solve * jnp.eye(
                parameter_count,
                dtype=solve_dtype,
            )
            chol = jsp.linalg.cho_factor(metric, lower=True)
            direction = jsp.linalg.cho_solve(chol, grad_solve)
        else:
            padded_parameter_count = (
                (parameter_count + device_count - 1) // device_count
            ) * device_count
            score_padded = jnp.pad(
                score_solve,
                ((0, 0), (0, padded_parameter_count - parameter_count)),
            )
            score_by_parameter = jax.lax.all_to_all(
                score_padded,
                axis_name=axis_name,
                split_axis=1,
                concat_axis=0,
                tiled=True,
            )
            kernel = jax.lax.psum(
                score_by_parameter @ score_by_parameter.T,
                axis_name=axis_name,
            )
            kernel = (kernel + kernel.T) / 2.0
            if local_kernel_null_vectors is not None:
                null_vectors = jax.lax.all_gather(
                    jnp.asarray(local_kernel_null_vectors, dtype=solve_dtype),
                    axis_name=axis_name,
                    axis=1,
                    tiled=True,
                )
                null_norm = jnp.linalg.norm(null_vectors, axis=1, keepdims=True)
                normalized_null = jnp.where(
                    null_norm > 0.0,
                    null_vectors
                    / jnp.maximum(
                        null_norm,
                        jnp.asarray(jnp.finfo(solve_dtype).tiny, dtype=solve_dtype),
                    ),
                    jnp.asarray(0.0, dtype=solve_dtype),
                )
                kernel = kernel + jnp.asarray(
                    kernel_projector_scale,
                    dtype=solve_dtype,
                ) * (normalized_null.T @ normalized_null)
            kernel = (kernel + kernel.T) / 2.0
            kernel = kernel + damping_solve * jnp.eye(
                sample_count,
                dtype=solve_dtype,
            )
            local_rhs = score_solve @ grad_solve
            rhs = jax.lax.all_gather(
                local_rhs,
                axis_name=axis_name,
                axis=0,
                tiled=True,
            )
            chol = jsp.linalg.cho_factor(kernel, lower=True)
            alpha = jsp.linalg.cho_solve(chol, rhs)
            alpha_start = jax.lax.axis_index(axis_name) * local_sample_count
            local_alpha = jax.lax.dynamic_slice_in_dim(
                alpha,
                alpha_start,
                local_sample_count,
                axis=0,
            )
            projected = jax.lax.psum(
                score_solve.T @ local_alpha,
                axis_name=axis_name,
            )
            direction = (grad_solve - projected) / damping_solve
        direction = direction.astype(original_dtype)
    return jnp.where(
        jnp.all(jnp.isfinite(direction)),
        direction,
        jnp.zeros_like(direction),
    )


def _solve_sr_direction_chunked(
    chunk_rows: tuple[int, ...],
    score_aug_chunk,
    grad_flat,
    damping,
    *,
    kernel_null_vectors=None,
    kernel_projector_scale=1.0,
):
    """Solve the SR system from score chunks without materializing all scores.

    Args:
        chunk_rows: Row count for each real-augmented score chunk.
        score_aug_chunk: Callable returning one real-augmented score chunk.
        grad_flat: Flattened objective gradient.
        damping: Positive SR damping added to the metric.
        kernel_null_vectors: Known left-null vectors of the centered score
            matrix, used to lift the Gram matrix null space.
        kernel_projector_scale: Positive eigenvalue assigned to those lifted
            null-space directions.

    Returns:
        Flattened preconditioned SR direction.

    Raises:
        ValueError: If no score chunks are provided.
    """
    if not chunk_rows:
        msg = "At least one SR score chunk is required."
        raise ValueError(msg)
    parameter_count = grad_flat.shape[0]
    sample_count = sum(int(rows) for rows in chunk_rows)
    original_dtype = grad_flat.dtype
    # Local x64 is deliberately enabled even when the rest of the workflow is
    # float32.  The Gram solve is small compared with score construction, and
    # this is the numerically sensitive part of SPRING.
    with _enable_x64(True):
        solve_dtype = jnp.float64
        grad_solve = grad_flat.astype(solve_dtype)
        damping_solve = jnp.asarray(damping, dtype=solve_dtype)
        score_chunks = tuple(
            score_aug_chunk(index).astype(solve_dtype)
            for index in range(len(chunk_rows))
        )
        if parameter_count <= sample_count:
            metric = jnp.zeros(
                (parameter_count, parameter_count),
                dtype=solve_dtype,
            )
            for score_aug in score_chunks:
                metric = metric + score_aug.T @ score_aug
            metric = (metric + metric.T) / 2.0
            metric = metric + damping_solve * jnp.eye(
                parameter_count,
                dtype=solve_dtype,
            )
            chol = jsp.linalg.cho_factor(metric, lower=True)
            direction = jsp.linalg.cho_solve(chol, grad_solve)
        else:
            row_blocks = []
            for row_score_aug in score_chunks:
                column_blocks = [
                    row_score_aug @ column_score_aug.T
                    for column_score_aug in score_chunks
                ]
                row_blocks.append(jnp.concatenate(column_blocks, axis=1))
            kernel = jnp.concatenate(row_blocks, axis=0)
            kernel = (kernel + kernel.T) / 2.0
            if kernel_null_vectors is not None:
                null_vectors = jnp.asarray(
                    kernel_null_vectors,
                    dtype=solve_dtype,
                )
                if null_vectors.ndim == 1:
                    null_vectors = null_vectors[None, :]
                null_norm = jnp.linalg.norm(null_vectors, axis=1, keepdims=True)
                normalized_null = jnp.where(
                    null_norm > 0.0,
                    null_vectors
                    / jnp.maximum(
                        null_norm,
                        jnp.asarray(jnp.finfo(solve_dtype).tiny, dtype=solve_dtype),
                    ),
                    jnp.asarray(0.0, dtype=solve_dtype),
                )
                kernel = kernel + jnp.asarray(
                    kernel_projector_scale,
                    dtype=solve_dtype,
                ) * (normalized_null.T @ normalized_null)
            kernel = (kernel + kernel.T) / 2.0
            kernel = kernel + damping_solve * jnp.eye(
                sample_count,
                dtype=solve_dtype,
            )
            rhs = jnp.concatenate(
                [score_aug @ grad_solve for score_aug in score_chunks],
                axis=0,
            )
            chol = jsp.linalg.cho_factor(kernel, lower=True)
            alpha = jsp.linalg.cho_solve(chol, rhs)
            projected = jnp.zeros_like(grad_solve)
            start = 0
            for score_aug, rows in zip(score_chunks, chunk_rows, strict=True):
                stop = start + int(rows)
                projected = projected + score_aug.T @ alpha[start:stop]
                start = stop
            direction = (grad_solve - projected) / damping_solve
        direction = direction.astype(original_dtype)
    return jnp.where(
        jnp.all(jnp.isfinite(direction)),
        direction,
        jnp.zeros_like(direction),
    )


def _apply_updates(params, updates):
    return jax.tree.map(operator.add, params, updates)


def _regularized_action_gradient(
    score,
    ratio,
    source_weight,
    *,
    reverse_kl_weight: float | jnp.ndarray,
    eps: float | jnp.ndarray,
    axis_name: str | None = None,
):
    """Return the PRL fidelity-minus-reverse-KL action-state gradient.

    ``ratio`` is ``barPsi / Phi`` and ``score`` is
    ``d log(barPsi) / d theta``.  Rescaling all ratios by their largest
    magnitude prevents overflow without changing the fidelity, reverse KL, or
    either gradient.
    """
    real_dtype = ratio.real.dtype
    eps_array = jnp.asarray(eps, dtype=real_dtype)
    finite_score = jnp.all(
        jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
        axis=1,
    )
    finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
    finite_weight = jnp.isfinite(source_weight) & (source_weight >= 0.0)
    valid = finite_score & finite_ratio & finite_weight
    score = jnp.where(valid[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
    ratio = jnp.where(valid, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
    source_weight = jnp.where(
        valid,
        source_weight,
        jnp.asarray(0.0, dtype=source_weight.dtype),
    )

    def global_sum(value, *, axis=None, keepdims=False):
        reduced = jnp.sum(value, axis=axis, keepdims=keepdims)
        if axis_name is not None:
            reduced = jax.lax.psum(reduced, axis_name=axis_name)
        return reduced

    def global_max(value):
        reduced = jnp.max(value)
        if axis_name is not None:
            reduced = jax.lax.pmax(reduced, axis_name=axis_name)
        return reduced

    safe_weight_sum = jnp.maximum(global_sum(source_weight), eps_array)
    phi_weight = source_weight / safe_weight_sum
    max_ratio_abs = global_max(jnp.where(phi_weight > 0.0, jnp.abs(ratio), 0.0))
    ratio_scale = jnp.where(
        max_ratio_abs > 0.0,
        max_ratio_abs,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    scaled_ratio = ratio / jax.lax.stop_gradient(ratio_scale)
    ratio_abs2 = jnp.abs(scaled_ratio) ** 2
    ratio_norm = global_sum(phi_weight * ratio_abs2)
    safe_ratio_norm = jnp.maximum(ratio_norm, eps_array)
    has_action_mass = jnp.isfinite(ratio_norm) & (ratio_norm > 0.0)
    psi_weight = phi_weight * ratio_abs2 / safe_ratio_norm

    score_mean = global_sum(
        psi_weight[:, None] * score,
        axis=0,
        keepdims=True,
    )
    centered_score = score - score_mean
    amplitude = global_sum(phi_weight * scaled_ratio)
    score_covariance = global_sum(
        phi_weight[:, None] * scaled_ratio[:, None] * centered_score,
        axis=0,
    )
    fidelity_gradient = 2.0 * jnp.real(
        jnp.conj(amplitude) * score_covariance / safe_ratio_norm
    )

    log_ratio_abs2 = 2.0 * jnp.log(jnp.maximum(jnp.abs(scaled_ratio), eps_array))
    log_ratio_mean = global_sum(psi_weight * log_ratio_abs2)
    centered_log_ratio = log_ratio_abs2 - log_ratio_mean
    reverse_kl_gradient = 2.0 * jnp.real(
        global_sum(
            psi_weight[:, None] * centered_score * centered_log_ratio[:, None],
            axis=0,
        )
    )
    combined_gradient = (
        fidelity_gradient
        - jnp.asarray(
            reverse_kl_weight,
            dtype=real_dtype,
        )
        * reverse_kl_gradient
    )
    reverse_kl = jnp.where(
        has_action_mass,
        jnp.maximum(
            log_ratio_mean - jnp.log(safe_ratio_norm),
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    fidelity = jnp.where(
        has_action_mass,
        jnp.clip(jnp.abs(amplitude) ** 2 / safe_ratio_norm, 0.0, 1.0),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    return (
        combined_gradient,
        fidelity_gradient,
        reverse_kl_gradient,
        psi_weight,
        centered_score,
        fidelity,
        reverse_kl,
    )


def _source_distillation_stats_from_log_ratios(
    log_ratio,
    source_weight,
    *,
    reverse_kl_weight: float | jnp.ndarray,
    axis_name: str | None = None,
) -> _SourceDistillationStats:
    """Evaluate normalized response/source overlap on ``pi_Phi`` samples.

    Returns:
        Scale-invariant fidelity, reverse-KL, ESS, and sample-health statistics.
    """
    real_dtype = jnp.real(log_ratio).dtype
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    finite = (
        jnp.isfinite(jnp.real(log_ratio))
        & jnp.isfinite(jnp.imag(log_ratio))
        & jnp.isfinite(source_weight)
        & (source_weight >= 0.0)
    )
    local_count = jnp.asarray(log_ratio.size, dtype=real_dtype)
    valid_count = jnp.sum(finite.astype(real_dtype))
    safe_real = jnp.where(finite & (source_weight > 0.0), jnp.real(log_ratio), -jnp.inf)
    log_scale = jnp.max(safe_real)
    if axis_name is not None:
        local_count = jax.lax.psum(local_count, axis_name=axis_name)
        valid_count = jax.lax.psum(valid_count, axis_name=axis_name)
        log_scale = jax.lax.pmax(log_scale, axis_name=axis_name)
    log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
    log_scale = jax.lax.stop_gradient(log_scale)
    ratio = jnp.where(
        finite,
        jnp.exp(log_ratio - log_scale),
        jnp.asarray(0.0, dtype=log_ratio.dtype),
    )
    weight = jnp.where(finite, source_weight, 0.0)

    def global_sum(value):
        total = jnp.sum(value)
        if axis_name is not None:
            total = jax.lax.psum(total, axis_name=axis_name)
        return total

    weight_sum = global_sum(weight)
    phi_weight = weight / jnp.maximum(weight_sum, eps)
    ratio_abs2 = jnp.abs(ratio) ** 2
    ratio_norm = global_sum(phi_weight * ratio_abs2)
    safe_ratio_norm = jnp.maximum(ratio_norm, eps)
    amplitude = global_sum(phi_weight * ratio)
    fidelity = jnp.clip(jnp.abs(amplitude) ** 2 / safe_ratio_norm, 0.0, 1.0)
    psi_weight = phi_weight * ratio_abs2 / safe_ratio_norm
    log_ratio_abs2 = 2.0 * jnp.log(jnp.maximum(jnp.abs(ratio), eps))
    reverse_kl = jnp.maximum(
        global_sum(psi_weight * log_ratio_abs2) - jnp.log(safe_ratio_norm),
        0.0,
    )
    psi_weight_sq_sum = global_sum(psi_weight**2)
    ess = 1.0 / jnp.maximum(psi_weight_sq_sum, eps)
    ess_fraction = ess / jnp.maximum(local_count, 1.0)
    invalid_fraction = 1.0 - valid_count / jnp.maximum(local_count, 1.0)
    has_mass = (weight_sum > 0.0) & (ratio_norm > 0.0)
    fidelity = jnp.where(has_mass, fidelity, 0.0)
    reverse_kl = jnp.where(has_mass, reverse_kl, jnp.inf)
    ess_fraction = jnp.where(has_mass, ess_fraction, 0.0)
    loss = (
        1.0 - fidelity + jnp.asarray(reverse_kl_weight, dtype=real_dtype) * reverse_kl
    )
    return _SourceDistillationStats(
        loss=loss,
        fidelity=fidelity,
        reverse_kl=reverse_kl,
        reweight_ess_fraction=ess_fraction,
        invalid_sample_fraction=invalid_fraction,
    )


def _finite_source_distillation_stats(stats: _SourceDistillationStats) -> bool:
    return all(
        np.isfinite(float(value))
        for value in (
            stats.loss,
            stats.fidelity,
            stats.reverse_kl,
            stats.reweight_ess_fraction,
            stats.invalid_sample_fraction,
        )
    )


def _spring_direction_chunked(
    chunk_rows: tuple[int, ...],
    score_aug_chunk,
    grad_flat,
    state: _SpringState,
    *,
    epsilon_scale: float | jnp.ndarray,
    damping_floor: float | jnp.ndarray,
    decay: float | jnp.ndarray,
    qfi_trace=None,
    kernel_null_vectors=None,
):
    """Solve the scale-invariant SPRING system and retain unscaled history.

    Returns:
        Unscaled direction, updated SPRING state, and absolute damping.
    """
    if qfi_trace is None:
        qfi_trace = jnp.asarray(0.0, dtype=grad_flat.dtype)
        for index in range(len(chunk_rows)):
            score_aug = score_aug_chunk(index)
            qfi_trace = qfi_trace + jnp.sum(score_aug**2)
    parameter_count = jnp.asarray(max(int(grad_flat.shape[0]), 1), grad_flat.dtype)
    mean_metric_diagonal = qfi_trace / parameter_count
    damping = jnp.maximum(
        jnp.asarray(epsilon_scale, dtype=grad_flat.dtype) * mean_metric_diagonal,
        jnp.asarray(damping_floor, dtype=grad_flat.dtype),
    )
    rhs = (
        grad_flat
        + damping
        * jnp.asarray(
            decay,
            dtype=grad_flat.dtype,
        )
        * state.previous_direction
    )
    direction = _solve_sr_direction_chunked(
        chunk_rows,
        score_aug_chunk,
        rhs,
        damping,
        kernel_null_vectors=kernel_null_vectors,
    )
    valid_system = (
        jnp.isfinite(qfi_trace)
        & (qfi_trace > 0.0)
        & jnp.isfinite(damping)
        & jnp.all(jnp.isfinite(grad_flat))
        & jnp.all(jnp.isfinite(state.previous_direction))
        & jnp.all(jnp.isfinite(direction))
    )
    direction = jnp.where(valid_system, direction, jnp.zeros_like(direction))
    return (
        direction,
        _SpringState(previous_direction=jax.lax.stop_gradient(direction)),
        damping,
    )


def _spring_direction_data_parallel(
    local_score_aug,
    grad_flat,
    state: _SpringState,
    *,
    epsilon_scale: float | jnp.ndarray,
    damping_floor: float | jnp.ndarray,
    decay: float | jnp.ndarray,
    device_count: int,
    qfi_trace,
    local_kernel_null_vectors=None,
    axis_name: str = parallel_jax.BATCH_AXIS_NAME,
):
    """Apply SPRING to a row-sharded score matrix.

    Returns:
        Replicated direction, replicated next history state, and damping.
    """
    parameter_count = jnp.asarray(max(int(grad_flat.shape[0]), 1), grad_flat.dtype)
    mean_metric_diagonal = qfi_trace / parameter_count
    damping = jnp.maximum(
        jnp.asarray(epsilon_scale, dtype=grad_flat.dtype) * mean_metric_diagonal,
        jnp.asarray(damping_floor, dtype=grad_flat.dtype),
    )
    rhs = (
        grad_flat
        + damping * jnp.asarray(decay, dtype=grad_flat.dtype) * state.previous_direction
    )
    direction = _solve_sr_direction_data_parallel(
        local_score_aug,
        rhs,
        damping,
        device_count=device_count,
        local_kernel_null_vectors=local_kernel_null_vectors,
        axis_name=axis_name,
    )
    valid_system = (
        jnp.isfinite(qfi_trace)
        & (qfi_trace > 0.0)
        & jnp.isfinite(damping)
        & jnp.all(jnp.isfinite(grad_flat))
        & jnp.all(jnp.isfinite(state.previous_direction))
        & jnp.all(jnp.isfinite(direction))
    )
    direction = jnp.where(valid_system, direction, jnp.zeros_like(direction))
    return (
        direction,
        _SpringState(previous_direction=jax.lax.stop_gradient(direction)),
        damping,
    )


def _direction_update_scale(
    direction,
    *,
    learning_rate: float,
    max_norm: float | None,
):
    """Return the scalar applied to an unscaled SPRING direction."""
    scale = jnp.asarray(learning_rate, dtype=direction.dtype)
    if max_norm is not None:
        direction_norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(
            scale,
            jnp.asarray(max_norm, dtype=direction.dtype)
            / (direction_norm + jnp.asarray(1e-12, dtype=direction.dtype)),
        )
    return scale


def _top_level_group_rms_and_norm(tree, group_name: str, *, dtype):
    """Return per-coordinate RMS and L2 norm for one top-level pytree group."""
    missing = jnp.asarray(jnp.nan, dtype=dtype)
    if group_name == "response":
        group = tree
    elif isinstance(tree, Mapping) and group_name in tree:
        group = tree[group_name]
    else:
        return missing, missing
    leaves = jax.tree_util.tree_leaves(group)
    element_count = sum(int(leaf.size) for leaf in leaves)
    if element_count == 0:
        return missing, missing
    squared_norm = jnp.asarray(0.0, dtype=dtype)
    for leaf in leaves:
        leaf_array = jnp.asarray(leaf)
        squared_norm = squared_norm + jnp.sum(jnp.abs(leaf_array) ** 2)
    norm = jnp.sqrt(jnp.maximum(jnp.real(squared_norm), 0.0))
    rms = norm / jnp.sqrt(jnp.asarray(element_count, dtype=dtype))
    return rms, norm


def _spring_optimizer_diagnostics(
    params,
    combined_gradient,
    fidelity_gradient,
    reverse_kl_gradient,
    direction,
    updates,
    previous_direction,
    *,
    reverse_kl_weight: float | jnp.ndarray,
    learning_rate: float,
    max_norm: float | None,
    damping,
    decay: float | jnp.ndarray,
    qfi_trace,
) -> _SpringOptimizerDiagnostics:
    """Summarize existing source-sampled SPRING tensors as scalar diagnostics.

    Returns:
        Scalar optimizer and top-level parameter-group diagnostics.
    """
    dtype = combined_gradient.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    combined_gradient_norm = jnp.linalg.norm(combined_gradient)
    fidelity_gradient_norm = jnp.linalg.norm(fidelity_gradient)
    weighted_reverse_kl_gradient = (
        jnp.asarray(reverse_kl_weight, dtype=dtype) * reverse_kl_gradient
    )
    weighted_reverse_kl_gradient_norm = jnp.linalg.norm(weighted_reverse_kl_gradient)
    gradient_product = fidelity_gradient_norm * weighted_reverse_kl_gradient_norm
    fidelity_kl_cosine = jnp.where(
        gradient_product > tiny,
        jnp.vdot(fidelity_gradient, weighted_reverse_kl_gradient).real
        / jnp.maximum(gradient_product, tiny),
        jnp.asarray(0.0, dtype=dtype),
    )
    component_gradient_norm = fidelity_gradient_norm + weighted_reverse_kl_gradient_norm
    gradient_cancellation_ratio = jnp.where(
        component_gradient_norm > tiny,
        combined_gradient_norm / jnp.maximum(component_gradient_norm, tiny),
        jnp.asarray(0.0, dtype=dtype),
    )

    direction_norm = jnp.linalg.norm(direction)
    learning_rate_array = jnp.asarray(learning_rate, dtype=dtype)
    update_scale = _direction_update_scale(
        direction,
        learning_rate=learning_rate,
        max_norm=max_norm,
    )
    update_norm = jnp.abs(update_scale) * direction_norm
    clip_factor = update_scale / learning_rate_array

    parameter_count = jnp.asarray(max(int(combined_gradient.size), 1), dtype=dtype)
    qfi_trace = jnp.asarray(qfi_trace, dtype=dtype)
    mean_metric_diagonal = qfi_trace / parameter_count
    damping = jnp.asarray(damping, dtype=dtype)
    history_rhs = (
        damping
        * jnp.asarray(decay, dtype=dtype)
        * jnp.asarray(previous_direction, dtype=dtype)
    )
    history_gradient_ratio = jnp.linalg.norm(history_rhs) / jnp.maximum(
        combined_gradient_norm,
        tiny,
    )

    _, unravel_fn = ravel_pytree(params)
    gradient_tree = unravel_fn(combined_gradient)
    group_gradient_rms = []
    group_update_norm = []
    for group_name in _SPRING_PARAMETER_GROUPS:
        gradient_rms, _ = _top_level_group_rms_and_norm(
            gradient_tree,
            group_name,
            dtype=dtype,
        )
        _, update_group_norm = _top_level_group_rms_and_norm(
            updates,
            group_name,
            dtype=dtype,
        )
        group_gradient_rms.append(gradient_rms)
        group_update_norm.append(update_group_norm)
    return _SpringOptimizerDiagnostics(
        available=jnp.asarray(True),
        combined_gradient_norm=combined_gradient_norm,
        fidelity_gradient_norm=fidelity_gradient_norm,
        weighted_reverse_kl_gradient_norm=weighted_reverse_kl_gradient_norm,
        fidelity_kl_cosine=fidelity_kl_cosine,
        gradient_cancellation_ratio=gradient_cancellation_ratio,
        direction_norm=direction_norm,
        update_norm=update_norm,
        clip_factor=clip_factor,
        damping=damping,
        mean_metric_diagonal=mean_metric_diagonal,
        history_gradient_ratio=history_gradient_ratio,
        parameter_group_gradient_rms=jnp.stack(group_gradient_rms),
        parameter_group_update_norm=jnp.stack(group_update_norm),
    )


def _scaled_direction_updates(
    params,
    direction,
    *,
    learning_rate: float,
    max_norm: float | None,
):
    _, unravel_fn = ravel_pytree(params)
    scale = _direction_update_scale(
        direction,
        learning_rate=learning_rate,
        max_norm=max_norm,
    )
    return unravel_fn(scale * direction)


def _regularized_loss(stats, reverse_kl_weight: float):
    return (
        1.0
        - stats.fidelity
        + jnp.asarray(
            reverse_kl_weight,
            dtype=stats.fidelity.dtype,
        )
        * stats.reverse_kl
    )


def _require_eligible_lit_checkpoint(
    stats,
    *,
    context: str,
) -> None:
    """Raise when a checkpoint is numerically invalid.

    Raises:
        RuntimeError: If the checkpoint is not eligible for propagation.
    """
    if _is_eligible_lit_checkpoint(stats):
        return
    raise RuntimeError(f"{context}; held-out statistics are non-finite or invalid.")


def _is_eligible_lit_checkpoint(stats) -> bool:
    """Return whether one held-out checkpoint is numerically admissible."""
    loss = float(jax.device_get(stats.loss))
    fidelity = float(jax.device_get(stats.fidelity))
    reverse_kl = float(jax.device_get(stats.reverse_kl))
    invalid = float(jax.device_get(stats.invalid_sample_fraction))
    finite = bool(
        np.all(
            np.isfinite(
                (
                    loss,
                    fidelity,
                    reverse_kl,
                    invalid,
                )
            )
        )
    )
    return finite and invalid <= 0.0


def _is_selectable_lit_checkpoint(
    stats,
    *,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    """Return whether a checkpoint is safe for selection and propagation."""
    return (
        _is_eligible_lit_checkpoint(stats)
        and _lit_stage_ess_failure(
            stats,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        is None
    )


def _is_better_lit_checkpoint(
    candidate,
    incumbent,
    *,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    """Prefer the highest-fidelity healthy held-out checkpoint.

    Returns:
        Whether the candidate should replace the incumbent.
    """

    def score(stats):
        loss = float(jax.device_get(stats.loss))
        fidelity = float(jax.device_get(stats.fidelity))
        reverse_kl = float(jax.device_get(stats.reverse_kl))
        valid = _is_selectable_lit_checkpoint(
            stats,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        return valid, fidelity, -loss, -reverse_kl

    candidate_score = score(candidate)
    incumbent_score = score(incumbent)
    if not candidate_score[0]:
        return False
    if not incumbent_score[0]:
        return True
    return candidate_score[1:] > incumbent_score[1:]


def _phase_angle(phase, dtype) -> jnp.ndarray:
    return jnp.asarray(jnp.angle(phase), dtype=dtype)


def _lit_error_monitor(
    *,
    fidelity: float,
    source_norm: float,
    normalization: complex,
    eta: float,
    error_d: float,
    error_d_valid: bool,
) -> float:
    """Apply the paper's leading-order error monitor, dividing by ``|N|`` once.

    Supplement Eq. (19) drops higher orders in ``1 - fidelity``.  The result is
    therefore useful as a convergence/systematic-error monitor near unit
    fidelity, but is not advertised as a rigorous upper bound at moderate
    fidelity.

    Returns:
        The finite bound monitor, or ``NaN`` when an input is invalid.
    """
    fidelity = float(fidelity)
    source_norm = float(source_norm)
    eta = float(eta)
    error_d = float(error_d)
    normalization_abs = float(abs(normalization))
    if (
        not bool(error_d_valid)
        or not np.isfinite(error_d)
        or error_d < 0.0
        or not np.isfinite(normalization_abs)
        or normalization_abs <= 0.0
        or not np.isfinite(fidelity)
        or not 0.0 < fidelity <= 1.0
        or not np.isfinite(source_norm)
        or source_norm < 0.0
        or not np.isfinite(eta)
        or eta <= 0.0
    ):
        return float("nan")
    phi_norm = float(np.sqrt(source_norm))
    return lit_error_bound(
        fidelity,
        phi_norm=phi_norm,
        normalization_abs=normalization_abs,
        eta=eta,
        d_factor=error_d,
    )


def _lit_stage_ess_failure(
    stats,
    *,
    min_reweight_ess_fraction: float,
) -> str | None:
    """Describe a held-out importance-sampling ESS failure, if any.

    Returns:
        A human-readable failure description, or None when the threshold passes.
    """
    ess_fraction = float(jax.device_get(stats.reweight_ess_fraction))
    if min_reweight_ess_fraction > 0.0 and (
        not np.isfinite(ess_fraction) or ess_fraction < float(min_reweight_ess_fraction)
    ):
        return (
            "ESS fraction="
            f"{ess_fraction:.6f} < required="
            f"{float(min_reweight_ess_fraction):.6f}"
        )
    return None


def _require_lit_stage_health(
    stats,
    *,
    min_reweight_ess_fraction: float,
    context: str,
) -> None:
    """Raise when a finite checkpoint misses the held-out ESS guard.

    Raises:
        RuntimeError: If the active ESS threshold is missed.
    """
    failure = _lit_stage_ess_failure(
        stats,
        min_reweight_ess_fraction=min_reweight_ess_fraction,
    )
    if failure is not None:
        raise RuntimeError(f"{context}; {failure}.")
