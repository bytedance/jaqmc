# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

r"""Single-width discrete-line fits for molecular Lorentz transforms.

This module implements the line analysis in main-text Eq. (local_lit) and
Supporting Information Eq. (single_width_fit):

.. math::

   \mathcal L_a(\omega,\eta)
   = \sum_n \frac{I_{an}}{(\omega_n-\omega)^2+\eta^2}
     + B_a(\omega),

with one fixed positive ``eta`` and a low-order polynomial ``B_a``.  The fit
minimizes the ordinary, unweighted sum of squared residuals.  It does not
reconstruct a continuum and does not use covariance weighting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import itemgetter
from typing import SupportsInt, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LITBlockStatistics:
    """Mean and correlated Monte Carlo uncertainty from matched blocks."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    block_count: int


@dataclass(frozen=True)
class LITInversionDiagnostics:
    """Diagnostics for the article-aligned ordinary least-squares line fit."""

    residual_norms: NDArray[np.float64]
    condition_numbers: NDArray[np.float64]
    effective_ranks: NDArray[np.int64]
    active_coefficients: NDArray[np.int64]
    solver_success: tuple[bool, ...]
    solver_messages: tuple[str, ...]
    solver_optimality: NDArray[np.float64]
    pole_fit_success: bool
    pole_fit_message: str
    pole_fit_iterations: int
    objective: float
    unique_eta_count: int
    underdetermined: bool
    underdetermined_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LITInversionResult:
    """Fitted line centers, transition strengths, and local backgrounds."""

    pole_energies: NDArray[np.float64]
    pole_energy_bounds: NDArray[np.float64]
    pole_strengths: NDArray[np.float64]
    background_coefficients: NDArray[np.float64]
    background_center: float
    background_scale: float
    fitted_lit: NDArray[np.float64]
    residual: NDArray[np.float64]
    diagnostics: LITInversionDiagnostics


@dataclass(frozen=True)
class LITPoleInitialization:
    """Data-driven initial line centers for a requested model order."""

    pole_energies: NDArray[np.float64]
    pole_energy_bounds: NDArray[np.float64]
    objective: float
    candidate_grid_points: int
    minimum_separation: float


@dataclass(frozen=True)
class _NNLSResult:
    x: NDArray[np.float64]
    success: bool
    message: str
    optimality: float
    iterations: int


@dataclass(frozen=True)
class _AxisSolve:
    strengths: NDArray[np.float64]
    background: NDArray[np.float64]
    fitted: NDArray[np.float64]
    residual: NDArray[np.float64]
    condition_number: float
    effective_rank: int
    active_coefficients: int
    success: bool
    message: str
    optimality: float
    objective: float


@dataclass(frozen=True)
class _PoleOptimization:
    x: NDArray[np.float64]
    objective: float
    success: bool
    message: str
    iterations: int


def lit_block_statistics(block_estimates: ArrayLike) -> LITBlockStatistics:
    """Estimate covariance of a signed-LIT mean from matched blocks."""
    values = np.asarray(block_estimates, dtype=np.float64)
    if values.ndim < 2:
        msg = (
            "block_estimates must have at least observation and block "
            f"dimensions, got {values.shape}"
        )
        raise ValueError(msg)
    block_count = int(values.shape[-1])
    if block_count < 2:
        msg = "at least two matched blocks are required for a covariance estimate"
        raise ValueError(msg)
    if values.shape[-2] < 1:
        msg = "block_estimates must contain at least one observation"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "block_estimates must contain only finite values"
        raise ValueError(msg)

    mean = np.mean(values, axis=-1)
    centered = values - mean[..., np.newaxis]
    covariance = np.einsum(
        "...ib,...jb->...ij",
        centered,
        centered,
        optimize=True,
    ) / (block_count * (block_count - 1))
    standard_error = np.sqrt(
        np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 0.0)
    )
    return LITBlockStatistics(
        mean=mean,
        covariance=covariance,
        standard_error=standard_error,
        block_count=block_count,
    )


def _observation_arrays(
    omega: ArrayLike,
    eta: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:
    omega_array = np.asarray(omega, dtype=np.float64)
    eta_array = np.asarray(eta, dtype=np.float64)
    try:
        omega_array, eta_array = np.broadcast_arrays(omega_array, eta_array)
    except ValueError as error:
        msg = (
            "omega and eta must be broadcast-compatible, got "
            f"{omega_array.shape} and {eta_array.shape}"
        )
        raise ValueError(msg) from error
    if omega_array.ndim == 0:
        omega_array = omega_array.reshape(1)
        eta_array = eta_array.reshape(1)
    if omega_array.size == 0:
        msg = "omega and eta must contain at least one observation"
        raise ValueError(msg)
    if not np.all(np.isfinite(omega_array)):
        msg = "omega must contain only finite values"
        raise ValueError(msg)
    if not np.all(np.isfinite(eta_array)) or np.any(eta_array <= 0.0):
        msg = "eta must contain only finite, positive values"
        raise ValueError(msg)
    return omega_array.ravel(), eta_array.ravel(), omega_array.shape


def _one_dimensional_finite_array(
    value: ArrayLike | None,
    name: str,
) -> NDArray[np.float64]:
    if value is None:
        return np.empty(0, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional, got {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _response_axes(
    signed_lit: ArrayLike,
    observation_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    response = np.asarray(signed_lit, dtype=np.float64)
    if response.shape == observation_shape:
        response = response[np.newaxis, ...]
    expected_suffix = observation_shape
    if (
        response.ndim != len(observation_shape) + 1
        or response.shape[1:] != expected_suffix
    ):
        msg = (
            "signed_lit must have the observation shape or one leading response "
            f"axis; got {response.shape} for observations {observation_shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(response)):
        msg = "signed_lit must contain only finite values"
        raise ValueError(msg)
    return response.reshape(response.shape[0], -1)


def _unique_eta_count(eta: NDArray[np.float64]) -> int:
    ordered = np.sort(eta)
    count = 1
    reference = float(ordered[0])
    for value in ordered[1:]:
        if not np.isclose(value, reference, rtol=1e-12, atol=1e-15):
            count += 1
            reference = float(value)
    return count


def _require_single_eta(eta: NDArray[np.float64]) -> float:
    count = _unique_eta_count(eta)
    if count != 1:
        msg = (
            "the article line fit requires exactly one fixed eta; "
            f"found {count} distinct widths"
        )
        raise ValueError(msg)
    return float(eta[0])


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    try:
        converted = int(cast(SupportsInt, value))
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"{name} must be an integer of at least {minimum}"
        raise ValueError(msg) from error
    if isinstance(value, (bool, np.bool_)) or converted != value or converted < minimum:
        msg = f"{name} must be an integer of at least {minimum}"
        raise ValueError(msg)
    return converted


def _validate_energies(energies: NDArray[np.float64]) -> None:
    if energies.size == 0:
        msg = "at least one initial pole energy is required"
        raise ValueError(msg)
    if np.any(np.diff(energies) <= 0.0):
        msg = "pole_energies must be strictly increasing"
        raise ValueError(msg)


def _pole_bounds(
    bounds: ArrayLike,
    pole_energies: NDArray[np.float64],
) -> NDArray[np.float64]:
    array = np.asarray(bounds, dtype=np.float64)
    if array.shape != (pole_energies.size, 2):
        msg = (
            "pole_energy_bounds must have shape (n_poles, 2), got "
            f"{array.shape} for {pole_energies.size} poles"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or np.any(array[:, 0] >= array[:, 1]):
        msg = "each pole energy bound must be finite and strictly increasing"
        raise ValueError(msg)
    if array.shape[0] > 1 and np.any(array[:-1, 1] >= array[1:, 0]):
        msg = "pole energy bounds must be ordered and non-overlapping"
        raise ValueError(msg)
    if np.any(pole_energies < array[:, 0]) or np.any(pole_energies > array[:, 1]):
        msg = "initial pole_energies must lie within pole_energy_bounds"
        raise ValueError(msg)
    return array


def _covariances(
    value: ArrayLike,
    n_axes: int,
    n_observations: int,
    *,
    relative_tolerance: float,
) -> NDArray[np.float64]:
    """Validate stored covariance used for archival input checks only."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (n_observations, n_observations):
        array = np.broadcast_to(
            array,
            (n_axes, n_observations, n_observations),
        )
    elif array.shape != (n_axes, n_observations, n_observations):
        msg = (
            "covariance must have shape (n_observations, n_observations) or "
            f"(n_axes, n_observations, n_observations), got {array.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "covariance must contain only finite values"
        raise ValueError(msg)
    validated = np.empty_like(array, dtype=np.float64)
    for axis, covariance_axis in enumerate(array):
        covariance_scale = float(np.max(np.abs(covariance_axis), initial=0.0))
        symmetry_tolerance = (
            max(relative_tolerance, 100.0 * np.finfo(np.float64).eps) * covariance_scale
        )
        if np.max(np.abs(covariance_axis - covariance_axis.T), initial=0.0) > (
            symmetry_tolerance
        ):
            msg = f"covariance for axis {axis} must be symmetric"
            raise ValueError(msg)
        symmetric = (covariance_axis + covariance_axis.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(symmetric)
        spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
        if eigenvalues[0] < -relative_tolerance * spectral_scale:
            msg = f"covariance for axis {axis} must be positive semidefinite"
            raise ValueError(msg)
        validated[axis] = symmetric
    return validated


def lit_pole_kernel(
    omega: ArrayLike,
    eta: ArrayLike,
    pole_energies: ArrayLike,
) -> NDArray[np.float64]:
    """Build ``1 / ((omega_n - omega)**2 + eta**2)`` for every line."""
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies")
    kernel = 1.0 / (
        (energies[np.newaxis, :] - omega_flat[:, np.newaxis]) ** 2
        + eta_flat[:, np.newaxis] ** 2
    )
    return kernel.reshape((*observation_shape, energies.size))


def _background_coordinates(
    omega: NDArray[np.float64],
    center: float | None = None,
    scale: float | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    actual_center = (
        float((np.min(omega) + np.max(omega)) / 2.0)
        if center is None
        else float(center)
    )
    actual_scale = (
        float((np.max(omega) - np.min(omega)) / 2.0) if scale is None else float(scale)
    )
    if not np.isfinite(actual_center):
        msg = "background_center must be finite"
        raise ValueError(msg)
    if not np.isfinite(actual_scale) or actual_scale <= 0.0:
        actual_scale = 1.0
    return (omega - actual_center) / actual_scale, actual_center, actual_scale


def _coefficient_matrix(
    value: ArrayLike | None,
    expected_columns: int,
    name: str,
) -> tuple[NDArray[np.float64] | None, bool]:
    if value is None:
        return None, False
    array = np.asarray(value, dtype=np.float64)
    was_one_dimensional = array.ndim == 1
    if was_one_dimensional:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != expected_columns:
        msg = (
            f"{name} must have shape ({expected_columns},) or "
            f"(n_axes, {expected_columns}), got {array.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array, was_one_dimensional


def forward_lit(
    omega: ArrayLike,
    eta: ArrayLike,
    *,
    pole_energies: ArrayLike,
    pole_strengths: ArrayLike,
    background_coefficients: ArrayLike | None = None,
    background_center: float | None = None,
    background_scale: float | None = None,
) -> NDArray[np.float64]:
    """Evaluate the article's local discrete-line model."""
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies")
    _validate_energies(energies)
    strengths, strengths_were_1d = _coefficient_matrix(
        pole_strengths,
        energies.size,
        "pole_strengths",
    )
    if strengths is None:
        raise ValueError("pole_strengths are required")
    if np.any(strengths < 0.0):
        msg = "pole_strengths must be nonnegative transition strengths"
        raise ValueError(msg)

    result = (
        strengths
        @ lit_pole_kernel(
            omega_flat,
            eta_flat,
            energies,
        )
        .reshape(omega_flat.size, energies.size)
        .T
    )
    background_was_1d = True
    if background_coefficients is not None:
        raw_background = np.asarray(background_coefficients, dtype=np.float64)
        if raw_background.ndim == 1:
            raw_background = raw_background[np.newaxis, :]
        else:
            background_was_1d = False
        if raw_background.ndim != 2 or raw_background.shape[1] < 1:
            msg = "background_coefficients must be one- or two-dimensional"
            raise ValueError(msg)
        if not np.all(np.isfinite(raw_background)):
            msg = "background_coefficients must contain only finite values"
            raise ValueError(msg)
        if raw_background.shape[0] not in (1, result.shape[0]):
            msg = "background and pole strengths use incompatible axis counts"
            raise ValueError(msg)
        if raw_background.shape[0] == 1 and result.shape[0] > 1:
            raw_background = np.broadcast_to(
                raw_background,
                (result.shape[0], raw_background.shape[1]),
            )
        x, _, _ = _background_coordinates(
            omega_flat,
            background_center,
            background_scale,
        )
        design = np.polynomial.polynomial.polyvander(
            x,
            raw_background.shape[1] - 1,
        )
        result = result + raw_background @ design.T

    result = result.reshape((result.shape[0], *observation_shape))
    if strengths_were_1d and background_was_1d:
        return result[0]
    return result


def oscillator_strengths(
    pole_energies: ArrayLike,
    pole_strengths: ArrayLike,
    axis_indices: ArrayLike = (0, 1, 2),
) -> NDArray[np.float64]:
    r"""Return ``f_0n = (2/3) omega_n sum_(a=x,y,z) I_an``."""
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies")
    strengths = np.asarray(pole_strengths, dtype=np.float64)
    indices = np.asarray(axis_indices)
    if strengths.shape != (3, energies.size):
        msg = (
            "pole_strengths must contain exactly x, y, and z rows with shape "
            f"(3, {energies.size}); got {strengths.shape}"
        )
        raise ValueError(msg)
    if indices.shape != (3,) or set(int(value) for value in indices) != {0, 1, 2}:
        msg = "axis_indices must identify exactly the Cartesian x, y, and z axes"
        raise ValueError(msg)
    if not np.all(np.isfinite(strengths)) or np.any(strengths < 0.0):
        msg = "pole_strengths must contain only finite, nonnegative values"
        raise ValueError(msg)
    if not np.all(np.isfinite(energies)) or np.any(energies < 0.0):
        msg = "pole_energies must contain only finite, nonnegative values"
        raise ValueError(msg)
    return (2.0 / 3.0) * energies * np.sum(strengths, axis=0)


def _kkt_optimality(
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> float:
    gradient = matrix.T @ (matrix @ coefficients - target)
    column_norms = np.linalg.norm(matrix, axis=0)
    target_norm = float(np.linalg.norm(target))
    gradient_scale = column_norms * max(target_norm, np.finfo(np.float64).tiny)
    normalized_gradient = np.divide(
        gradient,
        gradient_scale,
        out=np.zeros_like(gradient),
        where=gradient_scale > 0.0,
    )
    positive = coefficients > 0.0
    positive_violation = (
        float(np.max(np.abs(normalized_gradient[positive]), initial=0.0))
        if np.any(positive)
        else 0.0
    )
    bound_violation = float(np.max(-normalized_gradient[~positive], initial=0.0))
    return max(positive_violation, bound_violation)


def _nonnegative_least_squares(
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int | None,
) -> _NNLSResult:
    """Solve transition strengths with a NumPy Lawson-Hanson active set."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        msg = "solver_tolerance must be finite and positive"
        raise ValueError(msg)
    original_matrix = matrix
    original_target = target
    n_coefficients = matrix.shape[1]
    column_scales = np.linalg.norm(matrix, axis=0)
    safe_column_scales = np.where(column_scales > 0.0, column_scales, 1.0)
    matrix = matrix / safe_column_scales[np.newaxis, :]
    target_scale = float(np.linalg.norm(target))
    safe_target_scale = target_scale if target_scale > 0.0 else 1.0
    target = target / safe_target_scale
    iteration_limit = (
        max_iterations if max_iterations is not None else max(30 * n_coefficients, 1)
    )
    if iteration_limit < 1:
        msg = "solver_max_iterations must be positive or null"
        raise ValueError(msg)

    coefficients = np.zeros(n_coefficients, dtype=np.float64)
    passive = np.zeros(n_coefficients, dtype=bool)
    column_norms = np.linalg.norm(matrix, axis=0)
    target_norm = float(np.linalg.norm(target))
    dual_scale = column_norms * max(target_norm, np.finfo(np.float64).tiny)
    dual_tolerance = tolerance * dual_scale
    dual = matrix.T @ target
    iterations = 0
    success = True
    message = "KKT conditions satisfied"

    while np.any((~passive) & (dual > dual_tolerance)):
        if iterations >= iteration_limit:
            success = False
            message = "maximum NNLS iterations reached"
            break
        normalized_dual = np.divide(
            dual,
            dual_scale,
            out=np.full_like(dual, -np.inf),
            where=dual_scale > 0.0,
        )
        entering = int(np.argmax(np.where(~passive, normalized_dual, -np.inf)))
        passive[entering] = True
        while True:
            trial = np.zeros_like(coefficients)
            trial[passive] = np.linalg.lstsq(
                matrix[:, passive],
                target,
                rcond=None,
            )[0]
            coefficient_tolerance = tolerance * np.maximum(
                np.abs(trial),
                np.abs(coefficients),
            )
            nonpositive = passive & (trial <= 0.0)
            if not np.any(nonpositive):
                coefficients = trial
                break
            denominators = coefficients[nonpositive] - trial[nonpositive]
            valid = denominators > 0.0
            step = (
                float(np.min(coefficients[nonpositive][valid] / denominators[valid]))
                if np.any(valid)
                else 0.0
            )
            coefficients += step * (trial - coefficients)
            to_remove = passive & (coefficients <= coefficient_tolerance)
            coefficients[to_remove] = 0.0
            passive[to_remove] = False
            iterations += 1
            if iterations >= iteration_limit:
                success = False
                message = "maximum NNLS iterations reached"
                break
        if not success:
            break
        dual = matrix.T @ (target - matrix @ coefficients)
        iterations += 1

    coefficients = (
        np.maximum(coefficients, 0.0) * safe_target_scale / safe_column_scales
    )
    optimality = _kkt_optimality(
        original_matrix,
        original_target,
        coefficients,
    )
    optimality_limit = max(
        10.0 * tolerance,
        100.0 * np.finfo(np.float64).eps * max(original_matrix.shape, default=1),
    )
    if success and (not np.isfinite(optimality) or optimality > optimality_limit):
        success = False
        message = (
            "NNLS terminated with dimensionless KKT violation "
            f"{optimality:.3e} above {optimality_limit:.3e}"
        )
    return _NNLSResult(
        x=coefficients,
        success=success,
        message=message,
        optimality=optimality,
        iterations=iterations,
    )


def _matrix_condition(matrix: NDArray[np.float64]) -> tuple[float, int]:
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.where(column_norms > 0.0, column_norms, 1.0)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return np.inf, 0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == min(matrix.shape) and singular_values[-1] > 0.0
        else np.inf
    )
    return condition, rank


def _background_design(
    omega: NDArray[np.float64],
    order: int,
) -> tuple[NDArray[np.float64], float, float]:
    order = _positive_integer(order, "background_order", minimum=0)
    x, center, scale = _background_coordinates(omega)
    design = np.asarray(
        np.polynomial.polynomial.polyvander(x, order),
        dtype=np.float64,
    )
    return design, center, scale


def _solve_axes(
    omega: NDArray[np.float64],
    eta: NDArray[np.float64],
    response: NDArray[np.float64],
    energies: NDArray[np.float64],
    background_design: NDArray[np.float64],
    *,
    solver_tolerance: float,
    solver_max_iterations: int | None,
) -> list[_AxisSolve]:
    kernel = lit_pole_kernel(omega, eta, energies).reshape(
        omega.size,
        energies.size,
    )
    background_q = np.linalg.qr(background_design, mode="reduced")[0]
    residualized_kernel = kernel - background_q @ (background_q.T @ kernel)
    full_design = np.column_stack((kernel, background_design))
    condition_number, effective_rank = _matrix_condition(full_design)
    solves: list[_AxisSolve] = []
    for target in response:
        residualized_target = target - background_q @ (background_q.T @ target)
        solution = _nonnegative_least_squares(
            residualized_kernel,
            residualized_target,
            tolerance=solver_tolerance,
            max_iterations=solver_max_iterations,
        )
        strengths = solution.x
        background = np.linalg.lstsq(
            background_design,
            target - kernel @ strengths,
            rcond=None,
        )[0]
        fitted = kernel @ strengths + background_design @ background
        residual = fitted - target
        strength_scale = max(
            float(np.max(strengths, initial=0.0)),
            np.finfo(np.float64).tiny,
        )
        active_strengths = int(np.count_nonzero(strengths > 1e-10 * strength_scale))
        solves.append(
            _AxisSolve(
                strengths=strengths,
                background=background,
                fitted=fitted,
                residual=residual,
                condition_number=condition_number,
                effective_rank=effective_rank,
                active_coefficients=active_strengths + background_design.shape[1],
                success=solution.success,
                message=solution.message,
                optimality=solution.optimality,
                objective=float(residual @ residual),
            )
        )
    return solves


def _automatic_pole_bounds(
    energies: NDArray[np.float64],
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    if not lower < upper:
        msg = "the pole search window must have positive width"
        raise ValueError(msg)
    bounds = np.empty((energies.size, 2), dtype=np.float64)
    bounds[0, 0] = lower
    bounds[-1, 1] = upper
    for index, boundary in enumerate((energies[:-1] + energies[1:]) / 2.0):
        bounds[index, 1] = np.nextafter(boundary, -np.inf)
        bounds[index + 1, 0] = np.nextafter(boundary, np.inf)
    if energies.size == 1:
        bounds[0] = (lower, upper)
    return bounds


def _golden_coordinate_search(
    objective: Callable[[NDArray[np.float64]], float],
    current: NDArray[np.float64],
    current_objective: float,
    index: int,
    lower: float,
    upper: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], float]:
    def evaluate(value: float) -> tuple[NDArray[np.float64], float]:
        candidate = current.copy()
        candidate[index] = value
        return candidate, float(objective(candidate))

    scan = np.unique(np.concatenate((np.linspace(lower, upper, 9), [current[index]])))
    candidates = [evaluate(float(value)) for value in scan]
    best_candidate, best_objective = min(candidates, key=itemgetter(1))
    if current_objective <= best_objective:
        best_candidate, best_objective = current.copy(), current_objective
    best_index = int(np.argmin([value[1] for value in candidates]))
    left = float(scan[max(best_index - 1, 0)])
    right = float(scan[min(best_index + 1, scan.size - 1)])
    if left == right:
        return best_candidate, best_objective

    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    point_left = right - inverse_phi * (right - left)
    point_right = left + inverse_phi * (right - left)
    candidate_left, objective_left = evaluate(point_left)
    candidate_right, objective_right = evaluate(point_right)
    coordinate_tolerance = tolerance * max(1.0, abs(lower), abs(upper))
    for _ in range(80):
        if right - left <= coordinate_tolerance:
            break
        if objective_left <= objective_right:
            right = point_right
            point_right = point_left
            candidate_right, objective_right = candidate_left, objective_left
            point_left = right - inverse_phi * (right - left)
            candidate_left, objective_left = evaluate(point_left)
        else:
            left = point_left
            point_left = point_right
            candidate_left, objective_left = candidate_right, objective_right
            point_right = left + inverse_phi * (right - left)
            candidate_right, objective_right = evaluate(point_right)
    for candidate, candidate_objective in (
        (candidate_left, objective_left),
        (candidate_right, objective_right),
    ):
        if candidate_objective < best_objective:
            best_candidate, best_objective = candidate, candidate_objective
    return best_candidate, best_objective


def _fit_shared_pole_energies(
    objective: Callable[[NDArray[np.float64]], float],
    initial: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> _PoleOptimization:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        msg = "pole_fit_tolerance must be finite and positive"
        raise ValueError(msg)
    _positive_integer(max_iterations, "pole_fit_max_iterations")
    current = initial.copy()
    current_objective = float(objective(current))
    for iteration in range(1, max_iterations + 1):
        previous = current.copy()
        previous_objective = current_objective
        for index in range(current.size):
            current, current_objective = _golden_coordinate_search(
                objective,
                current,
                current_objective,
                index,
                float(bounds[index, 0]),
                float(bounds[index, 1]),
                tolerance,
            )
        movement = float(np.max(np.abs(current - previous), initial=0.0))
        energy_scale = max(float(np.max(np.abs(current), initial=0.0)), 1.0)
        objective_change = abs(previous_objective - current_objective)
        objective_scale = max(abs(previous_objective), abs(current_objective), 1.0)
        if movement <= tolerance * energy_scale and (
            objective_change <= tolerance * objective_scale
        ):
            return _PoleOptimization(
                x=current,
                objective=current_objective,
                success=True,
                message="bounded ordinary least-squares fit converged",
                iterations=iteration,
            )
    return _PoleOptimization(
        x=current,
        objective=current_objective,
        success=False,
        message="maximum pole-fit iterations reached",
        iterations=max_iterations,
    )


def _pole_candidate_grid(
    lower: float,
    upper: float,
    pole_count: int,
    candidate_grid_points: int | None,
) -> NDArray[np.float64]:
    if candidate_grid_points is None:
        count = max(257, 32 * pole_count + 1)
    else:
        count = _positive_integer(
            candidate_grid_points,
            "candidate_grid_points",
            minimum=max(4 * pole_count + 1, 17),
        )
    return np.linspace(lower, upper, count)


def _greedy_pole_candidates(
    candidates: NDArray[np.float64],
    pole_count: int,
    minimum_separation: float,
    objective: Callable[[NDArray[np.float64]], float | None],
) -> tuple[NDArray[np.float64], float]:
    selected: list[float] = []
    best_objective = np.inf
    for _ in range(pole_count):
        step_energy: float | None = None
        step_objective = np.inf
        for candidate in candidates:
            value = float(candidate)
            if any(abs(value - existing) < minimum_separation for existing in selected):
                continue
            trial = np.sort(np.asarray((*selected, value), dtype=np.float64))
            candidate_objective = objective(trial)
            if candidate_objective is not None and candidate_objective < step_objective:
                step_energy = value
                step_objective = candidate_objective
        if step_energy is None:
            msg = (
                "line initialization could not place all requested poles; reduce "
                "pole_count/minimum_separation or widen the search window"
            )
            raise RuntimeError(msg)
        selected.append(step_energy)
        selected.sort()
        best_objective = step_objective
    return np.asarray(selected), best_objective


def initialize_lit_poles(
    omega: ArrayLike,
    eta: ArrayLike,
    signed_lit: ArrayLike,
    *,
    pole_count: int,
    background_order: int = 0,
    energy_min: float | None = None,
    energy_max: float | None = None,
    candidate_grid_points: int | None = None,
    minimum_separation: float | None = None,
    solver_tolerance: float = 1e-10,
    solver_max_iterations: int | None = None,
) -> LITPoleInitialization:
    """Choose starting line centers using the article's ordinary objective."""
    pole_count = _positive_integer(pole_count, "pole_count")
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    _require_single_eta(eta_flat)
    response = _response_axes(signed_lit, observation_shape)
    background_design, _, _ = _background_design(omega_flat, background_order)
    lower = float(np.min(omega_flat) if energy_min is None else energy_min)
    upper = float(np.max(omega_flat) if energy_max is None else energy_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        msg = "energy_min and energy_max must define a finite increasing window"
        raise ValueError(msg)
    candidates = _pole_candidate_grid(
        lower,
        upper,
        pole_count,
        candidate_grid_points,
    )
    spacing = float(candidates[1] - candidates[0])
    if minimum_separation is None:
        separation = max(2.0 * float(eta_flat[0]), 2.0 * spacing)
    else:
        separation = float(minimum_separation)
        if not np.isfinite(separation) or separation <= 0.0:
            msg = "minimum_separation must be finite and positive"
            raise ValueError(msg)

    def objective(trial: NDArray[np.float64]) -> float | None:
        solves = _solve_axes(
            omega_flat,
            eta_flat,
            response,
            trial,
            background_design,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        )
        if not all(solve.success for solve in solves):
            return None
        return float(sum(solve.objective for solve in solves))

    energies, best_objective = _greedy_pole_candidates(
        candidates,
        pole_count,
        separation,
        objective,
    )
    bounds = _automatic_pole_bounds(energies, lower, upper)
    return LITPoleInitialization(
        pole_energies=energies,
        pole_energy_bounds=bounds,
        objective=best_objective,
        candidate_grid_points=candidates.size,
        minimum_separation=separation,
    )


def invert_signed_lit(
    omega: ArrayLike,
    eta: ArrayLike,
    signed_lit: ArrayLike,
    *,
    pole_energies: ArrayLike,
    pole_energy_bounds: ArrayLike | None = None,
    background_order: int = 0,
    pole_fit_tolerance: float = 1e-7,
    pole_fit_max_iterations: int = 200,
    solver_tolerance: float = 1e-10,
    solver_max_iterations: int | None = None,
) -> LITInversionResult:
    """Fit a fixed-width LIT to discrete Lorentzians plus a polynomial.

    Line centers are shared between response axes.  For every trial set of line
    centers, transition strengths and axis-specific polynomial coefficients are
    refitted by ordinary least squares.  Covariance arrays are intentionally
    not accepted because SI Eq. (single_width_fit) is unweighted.
    """
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    _require_single_eta(eta_flat)
    response = _response_axes(signed_lit, observation_shape)
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies").copy()
    _validate_energies(energies)
    order = _positive_integer(background_order, "background_order", minimum=0)
    background_design, background_center, background_scale = _background_design(
        omega_flat,
        order,
    )
    if pole_energy_bounds is None:
        bounds = _automatic_pole_bounds(
            energies,
            float(np.min(omega_flat)),
            float(np.max(omega_flat)),
        )
    else:
        bounds = _pole_bounds(pole_energy_bounds, energies)

    def objective(candidate: NDArray[np.float64]) -> float:
        solves = _solve_axes(
            omega_flat,
            eta_flat,
            response,
            candidate,
            background_design,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        )
        return float(sum(solve.objective for solve in solves))

    optimization = _fit_shared_pole_energies(
        objective,
        energies,
        bounds,
        tolerance=pole_fit_tolerance,
        max_iterations=pole_fit_max_iterations,
    )
    energies = optimization.x
    solves = _solve_axes(
        omega_flat,
        eta_flat,
        response,
        energies,
        background_design,
        solver_tolerance=solver_tolerance,
        solver_max_iterations=solver_max_iterations,
    )
    coefficient_count = energies.size + background_design.shape[1]
    underdetermined_reasons: list[str] = []
    if omega_flat.size <= coefficient_count:
        underdetermined_reasons.append(
            "the observation count does not exceed the per-axis coefficient count"
        )
    if any(solve.effective_rank < coefficient_count for solve in solves):
        underdetermined_reasons.append(
            "the line-plus-background design is rank deficient"
        )
    total_parameters = response.shape[0] * coefficient_count + energies.size
    if response.size <= total_parameters:
        underdetermined_reasons.append(
            "the total observation count does not exceed the fitted parameter count"
        )
    diagnostics = LITInversionDiagnostics(
        residual_norms=np.asarray(
            [np.linalg.norm(solve.residual) for solve in solves],
            dtype=np.float64,
        ),
        condition_numbers=np.asarray(
            [solve.condition_number for solve in solves],
            dtype=np.float64,
        ),
        effective_ranks=np.asarray(
            [solve.effective_rank for solve in solves],
            dtype=np.int64,
        ),
        active_coefficients=np.asarray(
            [solve.active_coefficients for solve in solves],
            dtype=np.int64,
        ),
        solver_success=tuple(solve.success for solve in solves),
        solver_messages=tuple(solve.message for solve in solves),
        solver_optimality=np.asarray(
            [solve.optimality for solve in solves],
            dtype=np.float64,
        ),
        pole_fit_success=optimization.success,
        pole_fit_message=optimization.message,
        pole_fit_iterations=optimization.iterations,
        objective=float(sum(solve.objective for solve in solves)),
        unique_eta_count=_unique_eta_count(eta_flat),
        underdetermined=bool(underdetermined_reasons),
        underdetermined_reasons=tuple(underdetermined_reasons),
    )
    return LITInversionResult(
        pole_energies=energies.copy(),
        pole_energy_bounds=bounds.copy(),
        pole_strengths=np.stack([solve.strengths for solve in solves]),
        background_coefficients=np.stack([solve.background for solve in solves]),
        background_center=background_center,
        background_scale=background_scale,
        fitted_lit=np.stack([solve.fitted for solve in solves]).reshape(
            (response.shape[0], *observation_shape)
        ),
        residual=np.stack([solve.residual for solve in solves]).reshape(
            (response.shape[0], *observation_shape)
        ),
        diagnostics=diagnostics,
    )


invert_lit = invert_signed_lit


__all__ = [
    "LITBlockStatistics",
    "LITInversionDiagnostics",
    "LITInversionResult",
    "LITPoleInitialization",
    "forward_lit",
    "initialize_lit_poles",
    "invert_lit",
    "invert_signed_lit",
    "lit_block_statistics",
    "lit_pole_kernel",
    "oscillator_strengths",
]
