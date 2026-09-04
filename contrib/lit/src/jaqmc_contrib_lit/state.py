# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Internal immutable state used by the LIT workflow."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import numpy as np

from jaqmc_contrib_lit.response import LITStats


class _SpringState(NamedTuple):
    """Unscaled SPRING direction associated with the current frequency."""

    previous_direction: jax.Array


class _SpringOptimizerDiagnostics(NamedTuple):
    """Low-cost scalar diagnostics for one source-sampled SPRING update."""

    available: jax.Array
    combined_gradient_norm: jax.Array
    fidelity_gradient_norm: jax.Array
    weighted_reverse_kl_gradient_norm: jax.Array
    fidelity_kl_cosine: jax.Array
    gradient_cancellation_ratio: jax.Array
    direction_norm: jax.Array
    update_norm: jax.Array
    clip_factor: jax.Array
    damping: jax.Array
    mean_metric_diagonal: jax.Array
    history_gradient_ratio: jax.Array
    parameter_group_gradient_rms: jax.Array
    parameter_group_update_norm: jax.Array


_SPRING_PARAMETER_GROUPS = ("response",)


class _LITUpdateCarry(NamedTuple):
    spring: _SpringState
    rng: jax.Array


class _SourceDistillationStats(NamedTuple):
    """Held-out source-overlap diagnostics for response initialization."""

    loss: jax.Array
    fidelity: jax.Array
    reverse_kl: jax.Array
    reweight_ess_fraction: jax.Array
    invalid_sample_fraction: jax.Array


class _ContinuationRecord(NamedTuple):
    omega: float
    optimized: bool
    selected_iteration: int
    stats: Any
    inherited_fidelity: float
    step: float
    bisections: int
    probe_accepted: bool
    min_step_override: bool


class _ContinuationCapacityDiagnostics(NamedTuple):
    """Point-budget forecast for one continuation proposal."""

    remaining_gap: float
    remaining_bridge_slots: int
    required_mean_step: float
    capacity_ratio: float


@dataclass
class _FidelityPlateauTracker:
    """Host-side early stopping on cumulative best held-out fidelity."""

    start_iteration: int
    patience_iterations: int
    min_delta: float
    reference_fidelity: float | None = None
    last_significant_iteration: int | None = None

    @property
    def enabled(self) -> bool:
        return self.patience_iterations > 0

    def observe(self, iteration: int, best_fidelity: float) -> bool:
        """Record the best fidelity and report whether it has plateaued.

        Returns:
            Whether the patience window has elapsed without a significant
            cumulative improvement.
        """
        if not self.enabled or iteration < self.start_iteration:
            return False
        if not np.isfinite(best_fidelity):
            return False
        if self.reference_fidelity is None:
            self.reference_fidelity = float(best_fidelity)
            self.last_significant_iteration = int(iteration)
            return False
        if best_fidelity > self.reference_fidelity + self.min_delta:
            self.reference_fidelity = float(best_fidelity)
            self.last_significant_iteration = int(iteration)
            return False
        if self.last_significant_iteration is None:
            return False
        return iteration - self.last_significant_iteration >= self.patience_iterations

    def defer(self, iteration: int) -> None:
        """Restart patience after an unhealthy held-out observation."""
        if (
            self.enabled
            and self.reference_fidelity is not None
            and iteration >= self.start_iteration
        ):
            self.last_significant_iteration = int(iteration)


class _ContinuationCheckpoint(NamedTuple):
    """Serializable latest-good state for one response-axis bridge chain."""

    schema_version: int
    state_fingerprint: str
    full_config_digest: str
    axis: int
    target_omega: float
    current_omega: float
    accepted_points: int
    ground_checkpoint_step: int
    ground_energy: float
    source_center: float
    source_norm: float
    response_parity: int
    response_params: Any
    rng: jax.Array
    current_stats: LITStats
    history_json: str
    warm_start_selected_iteration: int


class _ContinuationResumeState(NamedTuple):
    """Validated in-memory continuation state restored from one checkpoint."""

    response_params: Any
    rng: jax.Array
    current_stats: LITStats
    current_omega: float
    records: tuple[_ContinuationRecord, ...]
    warm_start_selected_iteration: int


class _AtomicParityResolution(NamedTuple):
    """Host-side atomic ground/response parity admission result."""

    ground_parity: int
    response_parity: int
    even_loss: float
    odd_loss: float
    selected_loss: float
