# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Frequency optimization and estimators for the molecular LIT workflow."""

from __future__ import annotations

import hashlib
import logging

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from jaqmc.data import BatchedData
from jaqmc.sampler.base import SamplePlan
from jaqmc.utils import parallel_jax
from jaqmc_contrib_lit.common import (
    _AXIS_NAMES,
    _three_component_override,
)
from jaqmc_contrib_lit.optimization import (
    _apply_updates,
    _is_better_lit_checkpoint,
    _is_selectable_lit_checkpoint,
    _log_spring_optimizer_diagnostics,
    _phase_angle,
    _regularized_action_gradient,
    _regularized_loss,
    _require_lit_stage_health,
    _scaled_direction_updates,
    _spring_direction_chunked,
    _spring_direction_data_parallel,
    _spring_optimizer_diagnostics,
)
from jaqmc_contrib_lit.pool import (
    _add_source_sums,
    _batched_data_chunks,
    _flatten_batched_tree,
    _indexed_batched_data_chunk,
    _lit_stats_from_source_sums,
    _merge_source_sums_across_devices,
    _replicate_across_local_devices,
    _shard_batched_data_across_local_devices,
    _shuffled_batched_data_chunk_index,
)
from jaqmc_contrib_lit.response import (
    LITSourceSums,
    ground_local_energy,
    local_action_ratio,
    molecular_electronic_dipole,
    restore_params_from_checkpoint,
    source_sampled_sums,
    stats_from_source_sums,
    weighted_complex_moments,
)
from jaqmc_contrib_lit.sector import (
    SourceSector,
    _log_source_center_projection,
    _project_source_center_to_invariant_subspace,
)
from jaqmc_contrib_lit.state import (
    _FidelityPlateauTracker,
    _LITUpdateCarry,
    _SpringState,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


class TrainingMixin:
    def _evaluate_lit_checkpoint(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        eval_pool,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega: float,
    ):
        stats = self._lit_stats_chunked(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            eval_pool,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=jnp.asarray(float(omega)),
        )
        physical_loss = _regularized_loss(
            stats,
            self.lit_config.solver.reverse_kl_weight,
        )
        return stats._replace(loss=physical_loss)

    def _optimize_lit_frequency(  # noqa: C901
        self,
        update_step,
        initial_params,
        train_pool,
        eval_pool,
        rng,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega: float,
        iterations: int,
        stage: str,
    ):
        """Optimize one frequency until its best held-out fidelity plateaus.

        Model selection and early stopping use one fixed evaluation pool.  The
        plateau clock starts only after the configured baseline iteration and
        resets when the cumulative best fidelity improves by more than the
        configured tolerance.  An unhealthy observation cannot be selected
        and restarts the patience clock.  The best numerically healthy,
        ESS-qualified checkpoint is returned even when the maximum budget is
        exhausted.

        Returns:
            Best parameters, their held-out statistics, selected iteration,
            and the next random key.

        Raises:
            ValueError: If the iteration budget is not positive.
            RuntimeError: If no held-out checkpoint satisfies numerical,
                source-covariance, and estimator-health requirements.
        """

        def evaluate(params):
            return self._evaluate_lit_checkpoint(
                response_apply,
                params,
                ground_logpsi,
                ground_params,
                eval_pool,
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=float(omega),
            )

        maximum_iterations = int(iterations)
        if maximum_iterations < 1:
            raise ValueError("NQS optimizer iterations must be positive.")
        plateau = _FidelityPlateauTracker(
            start_iteration=(self.lit_config.solver.plateau_start_iteration),
            patience_iterations=(self.lit_config.solver.plateau_patience_iterations),
            min_delta=self.lit_config.solver.plateau_min_delta,
        )
        forced_evaluations = {maximum_iterations}
        if plateau.enabled and 0 < plateau.start_iteration <= maximum_iterations:
            forced_evaluations.add(plateau.start_iteration)
        executed_iterations = maximum_iterations
        stop_reason = "max_budget"

        response_params = initial_params
        update_carry = update_step.init_carry(rng, response_params)
        minimum_ess = self.lit_config.continuation.ess_fraction_minimum
        initial_stats = jax.device_get(evaluate(response_params))
        latest_stats = initial_stats
        initial_fidelity = float(jax.device_get(initial_stats.fidelity))
        best_params = None
        best_stats = None
        best_iteration = -1
        if _is_selectable_lit_checkpoint(
            initial_stats,
            min_reweight_ess_fraction=minimum_ess,
        ):
            best_params = response_params
            best_stats = initial_stats
            best_iteration = 0
            if plateau.start_iteration == 0:
                plateau.observe(0, initial_fidelity)
        last_train_stats = None
        last_optimizer_diagnostics = None
        shuffle_seed = self._training_shuffle_seed(
            axis=axis,
            stage=stage,
            omega=float(omega),
        )
        for iteration in range(maximum_iterations):
            chunk_index = iteration
            if train_pool is not None:
                chunk_index = _shuffled_batched_data_chunk_index(
                    train_pool.batch_size,
                    self._lit_train_update_batch_size(),
                    iteration,
                    seed=shuffle_seed,
                )
            response_params, last_train_stats, update_carry = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(omega)),
                update_carry,
                chunk_index,
            )
            last_optimizer_diagnostics = getattr(
                update_step,
                "last_spring_optimizer_diagnostics",
                None,
            )
            completed = iteration + 1
            should_select = (
                completed % self.lit_config.solver.selection_interval == 0
                or completed in forced_evaluations
            )
            candidate_stats = None
            if should_select:
                candidate_stats = jax.device_get(evaluate(response_params))
                latest_stats = candidate_stats
                candidate_selectable = _is_selectable_lit_checkpoint(
                    candidate_stats,
                    min_reweight_ess_fraction=minimum_ess,
                )
                if not candidate_selectable:
                    plateau.defer(completed)
                if candidate_selectable and (
                    best_stats is None
                    or _is_better_lit_checkpoint(
                        candidate_stats,
                        best_stats,
                        min_reweight_ess_fraction=minimum_ess,
                    )
                ):
                    best_params = response_params
                    best_stats = candidate_stats
                    best_iteration = completed
            if (
                self.lit_config.solver.log_interval > 0
                and completed % self.lit_config.solver.log_interval == 0
            ):
                reported_stats = best_stats if best_stats is not None else latest_stats
                host_train_stats, host_optimizer_diagnostics = jax.device_get(
                    (last_train_stats, last_optimizer_diagnostics)
                )
                logger.info(
                    "axis=%s stage=%s omega=%.6f iter=%d train_loss=%.6e "
                    "train_fidelity=%.6f train_reverse_kl=%.6e train_ess=%.3f "
                    "best_iter=%d best_fidelity=%.6f best_reverse_kl=%.6e "
                    "best_ess=%.3f",
                    _AXIS_NAMES[axis],
                    stage,
                    float(omega),
                    completed,
                    float(host_train_stats.loss),
                    float(host_train_stats.fidelity),
                    float(host_train_stats.reverse_kl),
                    float(host_train_stats.reweight_ess_fraction),
                    best_iteration,
                    float(reported_stats.fidelity),
                    float(reported_stats.reverse_kl),
                    float(reported_stats.reweight_ess_fraction),
                )
                _log_spring_optimizer_diagnostics(
                    host_optimizer_diagnostics,
                    axis=axis,
                    stage=stage,
                    omega=float(omega),
                    iteration=completed,
                )
            if (
                candidate_stats is not None
                and best_stats is not None
                and _is_selectable_lit_checkpoint(
                    candidate_stats,
                    min_reweight_ess_fraction=minimum_ess,
                )
                and plateau.observe(
                    completed,
                    float(jax.device_get(best_stats.fidelity)),
                )
            ):
                executed_iterations = completed
                stop_reason = "fidelity_plateau"
                plateau_reference_fidelity = plateau.reference_fidelity
                plateau_last_significant_iteration = plateau.last_significant_iteration
                if (
                    plateau_reference_fidelity is None
                    or plateau_last_significant_iteration is None
                ):
                    msg = "plateau stop requires an initialized fidelity reference"
                    raise RuntimeError(msg)
                logger.info(
                    "axis=%s stage=%s omega=%.6f iter=%d action=plateau_stop "
                    "best_fidelity=%.6f reference_fidelity=%.6f "
                    "last_significant_iter=%d patience=%d min_delta=%.3e",
                    _AXIS_NAMES[axis],
                    stage,
                    float(omega),
                    completed,
                    float(best_stats.fidelity),
                    plateau_reference_fidelity,
                    plateau_last_significant_iteration,
                    plateau.patience_iterations,
                    plateau.min_delta,
                )
                break
        if best_stats is None or best_params is None:
            msg = (
                f"axis={_AXIS_NAMES[axis]} stage={stage} omega={float(omega):.6f} "
                "produced no healthy held-out checkpoint; ESS="
                f"{float(latest_stats.reweight_ess_fraction):.6f}, required ESS="
                f"{minimum_ess:.6f}."
            )
            raise RuntimeError(msg)
        _require_lit_stage_health(
            best_stats,
            min_reweight_ess_fraction=minimum_ess,
            context=(
                f"axis={_AXIS_NAMES[axis]} stage={stage} "
                f"omega={float(omega):.6f} failed held-out estimator health"
            ),
        )
        rng = update_carry.rng
        logger.info(
            "axis=%s stage=%s omega=%.6f selected_iter=%d/%d "
            "heldout_loss=%.6e fidelity=%.6f reverse_kl=%.6e "
            "ess=%.3f initial_fidelity=%.6f fidelity_gain=%+.6e "
            "stop_reason=%s required_ess=%.3f",
            _AXIS_NAMES[axis],
            stage,
            float(omega),
            best_iteration,
            executed_iterations,
            float(best_stats.loss),
            float(best_stats.fidelity),
            float(best_stats.reverse_kl),
            float(best_stats.reweight_ess_fraction),
            initial_fidelity,
            float(best_stats.fidelity) - initial_fidelity,
            stop_reason,
            minimum_ess,
        )
        return best_params, best_stats, best_iteration, rng

    def _warm_start_axis(
        self,
        update_step,
        response_params,
        train_pool,
        eval_pool,
        rng,
        **kwargs,
    ):
        if (
            self.lit_config.solver.warm_start_omega is None
            or self.lit_config.solver.warm_start_iterations <= 0
        ):
            return response_params, None, 0, rng
        result = self._optimize_lit_frequency(
            update_step,
            response_params,
            train_pool,
            eval_pool,
            rng,
            omega=float(self.lit_config.solver.warm_start_omega),
            iterations=self.lit_config.solver.warm_start_iterations,
            stage="warm_start",
            **kwargs,
        )
        if result[1] is not None:
            logger.info(
                "axis=%s warm_start omega=%.6f iterations=%d "
                "selected_iter=%d fidelity=%.6f reverse_kl=%.6e",
                _AXIS_NAMES[kwargs["axis"]],
                float(self.lit_config.solver.warm_start_omega),
                self.lit_config.solver.warm_start_iterations,
                result[2],
                float(result[1].fidelity),
                float(result[1].reverse_kl),
            )
        return result

    def _resolve_lit_ground_state(self, example, ground_rng):
        fallback_ground_params = self.wf.init_params(example, ground_rng)
        checkpoint_path = self.lit_config.ground.checkpoint_path or str(
            self.restore_path
        )
        try:
            checkpoint_step, ground_params = restore_params_from_checkpoint(
                checkpoint_path,
                fallback_ground_params,
            )
            logger.info(
                "Restored ground-state parameters from %s at step %d",
                checkpoint_path,
                checkpoint_step,
            )
        except FileNotFoundError:
            if not self.lit_config.ground.allow_untrained:
                raise
            checkpoint_step = -1
            ground_params = fallback_ground_params
            logger.warning(
                "No ground checkpoint found at %s; using untrained ground "
                "parameters because lit.ground.allow_untrained=true.",
                checkpoint_path,
            )
        return checkpoint_step, ground_params, self._ground_complex_logpsi

    def _ground_complex_logpsi(self, params, data) -> jnp.ndarray:
        phase, log_abs = self.wf.phase_logpsi(params, data)
        return log_abs + 1j * _phase_angle(phase, log_abs.dtype)

    def _resolve_ground_energy(
        self,
        ground_logpsi,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
    ):
        if self.lit_config.ground.energy is not None:
            return (
                float(self.lit_config.ground.energy),
                batched_data,
                sampler_state,
                rng,
            )
        energy_values = []
        for _ in range(max(1, self.lit_config.ground.energy_steps)):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )
            local = jax.vmap(
                lambda one: ground_local_energy(ground_logpsi, ground_params, one),
                in_axes=(batched_data.vmap_axis,),
            )(batched_data.data)
            energy_values.append(float(jnp.mean(local)))
        return float(np.mean(energy_values)), batched_data, sampler_state, rng

    def _estimate_vector_source_stats(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        source_sector: SourceSector | None = None,
    ):
        center_override = _three_component_override(
            self.lit_config.source.center_override,
            name="lit.source.center_override",
        )
        norm_override = _three_component_override(
            self.lit_config.source.norm_override,
            name="lit.source.norm_override",
            positive=True,
        )
        electron_count = int(batched_data.data.electrons.shape[-2])
        if center_override is not None and norm_override is not None:
            center = _project_source_center_to_invariant_subspace(
                center_override,
                source_sector,
                electron_count=electron_count,
                tolerance=float(self.lit_config.ansatz.sector_tolerance),
            )
            _log_source_center_projection(
                center_override,
                center,
                source_sector,
            )
            return (
                center,
                norm_override,
                batched_data,
                sampler_state,
                rng,
            )
        mean_values = []
        mean_square_values = []
        for _ in range(max(1, self.lit_config.source.center_steps)):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )
            dipole = jax.vmap(
                lambda one: -jnp.sum(one.electrons, axis=0),
                in_axes=(batched_data.vmap_axis,),
            )(batched_data.data)
            mean_values.append(np.asarray(jnp.mean(dipole, axis=0)))
            mean_square_values.append(np.asarray(jnp.mean(dipole**2, axis=0)))
        mean = np.mean(mean_values, axis=0, dtype=np.float64)
        center = np.array(mean, copy=True)
        if center_override is not None:
            center = center_override
        unprojected_center = np.array(center, copy=True)
        center = _project_source_center_to_invariant_subspace(
            center,
            source_sector,
            electron_count=electron_count,
            tolerance=float(self.lit_config.ansatz.sector_tolerance),
        )
        _log_source_center_projection(
            unprojected_center,
            center,
            source_sector,
        )
        variance = (
            np.mean(mean_square_values, axis=0, dtype=np.float64)
            - 2.0 * center * mean
            + center**2
        )
        norm = np.maximum(variance, 1e-12)
        if norm_override is not None:
            norm = norm_override
        return center, norm, batched_data, sampler_state, rng

    def _make_source_log_amplitude(
        self,
        axis: int,
        source_center: float,
        ground_logpsi,
    ):
        floor = float(self.lit_config.source.floor)

        def log_amplitude(params, data):
            source = molecular_electronic_dipole(data, axis) - source_center
            return ground_logpsi(params, data) + jnp.log(
                jnp.maximum(jnp.abs(source), floor)
            )

        return log_amplitude

    def _make_lit_update_step(
        self,
        response_apply,
        ground_params,
        ground_logpsi,
        ground_energy: float,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
    ):
        data_parallel = self._lit_data_parallel_enabled()
        data_parallel_device_count = jax.local_device_count()
        data_parallel_ground_params = (
            _replicate_across_local_devices(ground_params)
            if data_parallel
            else ground_params
        )

        def source_update_impl(
            response_params,
            local_ground_params,
            batched_data,
            spring_previous,
            omega,
        ):
            if data_parallel:
                (
                    stats,
                    updates,
                    spring_state,
                    _,
                    optimizer_diagnostics,
                ) = self._source_sr_stats_and_updates_data_parallel(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    local_ground_params,
                    batched_data,
                    spring_state=_SpringState(spring_previous),
                    axis=axis,
                    source_center=source_center,
                    source_norm=source_norm,
                    ground_energy=ground_energy,
                    omega=omega,
                    device_count=data_parallel_device_count,
                )
            else:
                (
                    stats,
                    updates,
                    spring_state,
                    _,
                    optimizer_diagnostics,
                ) = self._source_sr_stats_and_updates(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    local_ground_params,
                    batched_data,
                    spring_state=_SpringState(spring_previous),
                    axis=axis,
                    source_center=source_center,
                    source_norm=source_norm,
                    ground_energy=ground_energy,
                    omega=omega,
                )
            response_params = _apply_updates(response_params, updates)
            loss = _regularized_loss(
                stats,
                self.lit_config.solver.reverse_kl_weight,
            )
            return (
                response_params,
                stats._replace(loss=loss),
                spring_state.previous_direction,
                optimizer_diagnostics,
            )

        source_update_kernel = None if data_parallel else jax.jit(source_update_impl)

        def update(
            response_params,
            batched_data,
            omega,
            update_carry,
            batch_index: int = 0,
        ):
            nonlocal source_update_kernel
            update_batch = _indexed_batched_data_chunk(
                batched_data,
                self._lit_train_update_batch_size(),
                batch_index,
            )
            if source_update_kernel is None:
                source_update_kernel = self._data_parallel_source_update_kernel(
                    source_update_impl,
                    update_batch,
                )
            kernel_response_params = response_params
            kernel_ground_params = data_parallel_ground_params
            kernel_batch = update_batch
            kernel_spring_previous = update_carry.spring.previous_direction
            kernel_omega = omega
            if data_parallel:
                kernel_response_params = _replicate_across_local_devices(
                    kernel_response_params
                )
                kernel_batch = _shard_batched_data_across_local_devices(kernel_batch)
                kernel_spring_previous = _replicate_across_local_devices(
                    kernel_spring_previous
                )
                kernel_omega = _replicate_across_local_devices(kernel_omega)
            (
                response_params,
                stats,
                spring_previous,
                optimizer_diagnostics,
            ) = source_update_kernel(
                kernel_response_params,
                kernel_ground_params,
                kernel_batch,
                kernel_spring_previous,
                kernel_omega,
            )
            update.last_spring_optimizer_diagnostics = optimizer_diagnostics
            return (
                response_params,
                stats,
                update_carry._replace(spring=_SpringState(spring_previous)),
            )

        update.init_carry = self._init_lit_update_carry
        update.last_spring_optimizer_diagnostics = None
        return update

    def _source_sr_stats_and_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        spring_state: _SpringState,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        score, ratio, source_weight, source_sums = (
            self._source_sampled_action_scores_and_sums(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
        )
        stats = stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )
        updates, spring_state, damping, optimizer_diagnostics = (
            self._weighted_sr_updates_from_scores(
                response_params,
                score,
                ratio,
                source_weight,
                spring_state,
            )
        )
        return (
            stats,
            updates,
            spring_state,
            damping,
            optimizer_diagnostics,
        )

    def _source_sr_stats_and_updates_data_parallel(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        spring_state: _SpringState,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
        device_count: int,
    ):
        """Evaluate one exact global source update from local batch shards.

        Returns:
            Global statistics, replicated updates and SPRING state, damping,
            and optimizer diagnostics.
        """
        score, ratio, source_weight, local_source_sums = (
            self._source_sampled_action_scores_and_sums(
                response_apply,
                parallel_jax.pvary(response_params),
                ground_logpsi,
                ground_params,
                batched_data,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
        )
        source_sums = _merge_source_sums_across_devices(local_source_sums)
        stats = stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )
        updates, spring_state, damping, optimizer_diagnostics = (
            self._weighted_sr_updates_from_scores_data_parallel(
                response_params,
                score,
                ratio,
                source_weight,
                spring_state,
                device_count=device_count,
            )
        )
        return (
            stats,
            updates,
            spring_state,
            damping,
            optimizer_diagnostics,
        )

    def _weighted_sr_updates_from_scores_data_parallel(
        self,
        response_params,
        score,
        ratio,
        source_weight,
        spring_state: _SpringState,
        *,
        device_count: int,
    ):
        """Return a replicated SPRING update from row-sharded action scores."""
        (
            grad_flat,
            fidelity_gradient,
            reverse_kl_gradient,
            psi_weight,
            centered_score,
            _,
            _,
        ) = _regularized_action_gradient(
            score,
            ratio,
            source_weight,
            reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
            eps=self.lit_config.solver.sr_score_epsilon,
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )
        weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
        local_score_aug = jnp.concatenate(
            [weighted_score.real, weighted_score.imag],
            axis=0,
        )
        qfi_trace = jax.lax.psum(
            jnp.sum(local_score_aug**2),
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )
        centering_null = jnp.sqrt(psi_weight)
        zero_null = jnp.zeros_like(centering_null)
        local_kernel_null_vectors = jnp.stack(
            [
                jnp.concatenate([centering_null, zero_null]),
                jnp.concatenate([zero_null, centering_null]),
            ]
        )
        previous_direction = spring_state.previous_direction
        direction, spring_state, damping = _spring_direction_data_parallel(
            local_score_aug,
            grad_flat,
            spring_state,
            epsilon_scale=self.lit_config.solver.spring_epsilon,
            damping_floor=self.lit_config.solver.spring_damping_floor,
            decay=self.lit_config.solver.spring_decay,
            device_count=device_count,
            qfi_trace=qfi_trace,
            local_kernel_null_vectors=local_kernel_null_vectors,
        )
        updates = _scaled_direction_updates(
            response_params,
            direction,
            learning_rate=self.lit_config.solver.learning_rate,
            max_norm=self.lit_config.solver.sr_max_norm,
        )
        optimizer_diagnostics = _spring_optimizer_diagnostics(
            response_params,
            grad_flat,
            fidelity_gradient,
            reverse_kl_gradient,
            direction,
            updates,
            previous_direction,
            reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
            learning_rate=self.lit_config.solver.learning_rate,
            max_norm=self.lit_config.solver.sr_max_norm,
            damping=damping,
            decay=self.lit_config.solver.spring_decay,
            qfi_trace=qfi_trace,
        )
        return updates, spring_state, damping, optimizer_diagnostics

    def _weighted_sr_updates_from_scores(
        self,
        response_params,
        score,
        ratio,
        source_weight,
        spring_state: _SpringState,
    ):
        (
            grad_flat,
            fidelity_gradient,
            reverse_kl_gradient,
            psi_weight,
            centered_score,
            _,
            _,
        ) = _regularized_action_gradient(
            score,
            ratio,
            source_weight,
            reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
            eps=self.lit_config.solver.sr_score_epsilon,
        )
        weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
        score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)
        qfi_trace = jnp.sum(score_aug**2)
        centering_null = jnp.sqrt(psi_weight)
        zero_null = jnp.zeros_like(centering_null)
        kernel_null_vectors = jnp.stack(
            [
                jnp.concatenate([centering_null, zero_null]),
                jnp.concatenate([zero_null, centering_null]),
            ]
        )
        previous_direction = spring_state.previous_direction
        direction, spring_state, damping = _spring_direction_chunked(
            (score_aug.shape[0],),
            lambda _: score_aug,
            grad_flat,
            spring_state,
            epsilon_scale=self.lit_config.solver.spring_epsilon,
            damping_floor=self.lit_config.solver.spring_damping_floor,
            decay=self.lit_config.solver.spring_decay,
            qfi_trace=qfi_trace,
            kernel_null_vectors=kernel_null_vectors,
        )
        updates = _scaled_direction_updates(
            response_params,
            direction,
            learning_rate=self.lit_config.solver.learning_rate,
            max_norm=self.lit_config.solver.sr_max_norm,
        )
        optimizer_diagnostics = _spring_optimizer_diagnostics(
            response_params,
            grad_flat,
            fidelity_gradient,
            reverse_kl_gradient,
            direction,
            updates,
            previous_direction,
            reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
            learning_rate=self.lit_config.solver.learning_rate,
            max_norm=self.lit_config.solver.sr_max_norm,
            damping=damping,
            decay=self.lit_config.solver.spring_decay,
            qfi_trace=qfi_trace,
        )
        return updates, spring_state, damping, optimizer_diagnostics

    def _source_sampled_action_scores_and_sums(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        data = batched_data.data
        score_eps = float(self.lit_config.solver.sr_score_epsilon)

        def action_score_and_aux(params, one):
            def split_log_action(local_params):
                local_action, local_response_ratio, local_eloc_response = (
                    local_action_ratio(
                        response_apply,
                        local_params,
                        ground_logpsi,
                        ground_params,
                        one,
                        ground_energy=ground_energy,
                        omega=omega,
                        eta=self.lit_config.eta,
                    )
                )
                safe_action = jnp.where(
                    jnp.abs(local_action) > score_eps,
                    local_action,
                    jnp.asarray(score_eps, dtype=local_action.real.dtype) + 0j,
                )
                value = jnp.log(safe_action)
                return jnp.stack([jnp.real(value), jnp.imag(value)]), (
                    local_action,
                    local_response_ratio,
                    local_eloc_response,
                )

            jac, aux = jax.jacrev(split_log_action, has_aux=True)(params)
            score_tree = jax.tree.map(lambda leaf: leaf[0] + 1j * leaf[1], jac)
            action, response_ratio, eloc_response = aux
            return action, response_ratio, eloc_response, score_tree

        action, response_ratio, eloc_response, score_tree = jax.vmap(
            lambda one: action_score_and_aux(response_params, one),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        score = _flatten_batched_tree(score_tree, action.shape[0])
        dipole = jax.vmap(
            lambda one: molecular_electronic_dipole(one, axis),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
        floor = jnp.asarray(self.lit_config.source.floor, dtype=dipole.dtype)

        stats_eps = jnp.asarray(1e-12, dtype=dipole.dtype)
        stats_sampled_source = jnp.maximum(jnp.abs(source), floor)
        stats_source_weight = (
            jnp.abs(source) / jnp.maximum(stats_sampled_source, stats_eps)
        ) ** 2
        base_finite_sums = (
            jnp.isfinite(jnp.real(action))
            & jnp.isfinite(jnp.imag(action))
            & jnp.isfinite(jnp.real(response_ratio))
            & jnp.isfinite(jnp.imag(response_ratio))
            & jnp.isfinite(source)
            & jnp.isfinite(stats_source_weight)
        )
        safe_source_stats = jnp.where(
            jnp.abs(source) > stats_eps,
            source,
            stats_eps * jnp.where(source < 0, -1.0, 1.0),
        )
        raw_stats_ratio = action / safe_source_stats
        finite_stats_ratio = jnp.isfinite(jnp.real(raw_stats_ratio)) & jnp.isfinite(
            jnp.imag(raw_stats_ratio)
        )
        finite_sums = base_finite_sums & finite_stats_ratio
        stats_action = jnp.where(
            finite_sums,
            action,
            jnp.asarray(0.0, dtype=action.dtype),
        )
        stats_response_ratio = jnp.where(
            finite_sums,
            response_ratio,
            jnp.asarray(0.0, dtype=response_ratio.dtype),
        )
        stats_source_weight = jnp.where(
            finite_sums,
            stats_source_weight,
            jnp.asarray(0.0, dtype=stats_source_weight.dtype),
        )
        stats_ratio = jnp.where(
            finite_sums,
            raw_stats_ratio,
            jnp.asarray(0.0, dtype=raw_stats_ratio.dtype),
        )
        stats_ratio_abs = jnp.where(
            stats_source_weight > 0.0,
            jnp.abs(stats_ratio),
            0.0,
        )
        max_stats_ratio_abs = jnp.max(stats_ratio_abs)
        stats_ratio_scale = jnp.where(
            max_stats_ratio_abs > 0.0,
            max_stats_ratio_abs,
            jnp.asarray(1.0, dtype=stats_ratio_abs.dtype),
        )
        scaled_stats_ratio = stats_ratio / jax.lax.stop_gradient(stats_ratio_scale)
        scaled_stats_ratio_abs2 = jnp.abs(scaled_stats_ratio) ** 2
        psi_weight_unnormalized = stats_source_weight * scaled_stats_ratio_abs2
        log_ratio_abs2 = 2.0 * jnp.log(
            jnp.maximum(jnp.abs(scaled_stats_ratio), stats_eps)
        )
        shift = jnp.asarray(omega, dtype=stats_response_ratio.real.dtype) + 1j * (
            jnp.asarray(self.lit_config.eta, dtype=stats_response_ratio.real.dtype)
        )
        hbar_response_ratio = stats_action + shift * stats_response_ratio
        response_over_source = stats_response_ratio / safe_source_stats
        hbar_over_source = hbar_response_ratio / safe_source_stats
        response_over_source_moments = weighted_complex_moments(
            response_over_source,
            stats_source_weight,
        )
        hbar_over_source_moments = weighted_complex_moments(
            hbar_over_source,
            stats_source_weight,
        )
        eloc_finite = jnp.isfinite(jnp.real(eloc_response))
        eloc_response = jnp.where(
            eloc_finite,
            eloc_response,
            jnp.asarray(0.0, dtype=eloc_response.dtype),
        )
        sample_count = jnp.asarray(action.shape[0], dtype=stats_source_weight.dtype)
        source_sums = LITSourceSums(
            sample_count=sample_count,
            weight_sum=jnp.sum(stats_source_weight),
            valid_sample_count=jnp.sum(finite_sums),
            ratio_scale=stats_ratio_scale,
            ratio_sum=jnp.sum(stats_source_weight * scaled_stats_ratio),
            ratio_abs2_sum=jnp.sum(stats_source_weight * scaled_stats_ratio_abs2),
            psi_weight_sum=jnp.sum(psi_weight_unnormalized),
            psi_weight_sq_sum=jnp.sum(psi_weight_unnormalized**2),
            psi_log_ratio_abs2_sum=jnp.sum(psi_weight_unnormalized * log_ratio_abs2),
            response_conj_over_source_sum=jnp.sum(
                stats_source_weight * jnp.conj(stats_response_ratio) / safe_source_stats
            ),
            ground_energy_sum=jnp.real(jnp.sum(eloc_response)),
            response_over_source_moments=response_over_source_moments,
            hbar_over_source_moments=hbar_over_source_moments,
            psi_weight_max=jnp.max(psi_weight_unnormalized),
        )

        safe_source_score = jnp.where(
            jnp.abs(source) > score_eps,
            source,
            jnp.asarray(score_eps, dtype=source.dtype)
            * jnp.where(source < 0, -1.0, 1.0),
        )
        score_sampled_source = jnp.maximum(jnp.abs(source), floor)
        source_weight = (
            jnp.abs(source) / jnp.maximum(score_sampled_source, score_eps)
        ) ** 2
        ratio = action / safe_source_score
        finite_score = jnp.all(
            jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
            axis=1,
        )
        finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
        finite_weight = jnp.isfinite(source_weight)
        finite = finite_score & finite_ratio & finite_weight
        score = jnp.where(finite[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
        ratio = jnp.where(finite, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
        source_weight = jnp.where(
            finite,
            source_weight,
            jnp.asarray(0.0, dtype=source_weight.dtype),
        )
        return score, ratio, source_weight, source_sums

    def _lit_chunk_sums_kernel(
        self,
        response_apply,
        ground_logpsi,
        batched_data,
        *,
        axis: int,
    ):
        """Return a persistent compiled kernel for held-out source sums.

        JAX caches a compiled executable by jitted callable identity.  Creating
        the callable inside :meth:`_lit_stats_chunked` therefore retraced and
        recompiled the same expensive local-action kernel at every checkpoint
        selection.  Cache one callable per response/ground closure and static
        dipole axis; all numerical values remain dynamic arguments so the same
        executable is reused across checkpoints and frequencies.

        Returns:
            A jitted callable that evaluates additive LIT source sums for
            one fixed-shape chunk.
        """
        cache = getattr(self, "_lit_chunk_sums_kernel_cache", None)
        if cache is None:
            cache = {}
            self._lit_chunk_sums_kernel_cache = cache
        data_parallel = self._lit_data_parallel_enabled()
        device_count = jax.local_device_count() if data_parallel else 1
        key = (
            id(response_apply),
            id(ground_logpsi),
            int(axis),
            data_parallel,
            device_count,
        )
        cached = cache.get(key)
        if (
            cached is not None
            and cached[0] is response_apply
            and cached[1] is ground_logpsi
        ):
            return cached[2]

        def chunk_sums_impl(
            local_params,
            local_ground_params,
            chunk,
            local_source_center,
            local_ground_energy,
            local_omega,
            local_eta,
            local_source_floor,
        ):
            sums = source_sampled_sums(
                response_apply,
                local_params,
                ground_logpsi,
                local_ground_params,
                chunk,
                axis=axis,
                source_center=local_source_center,
                ground_energy=local_ground_energy,
                omega=local_omega,
                eta=local_eta,
                source_floor=local_source_floor,
            )
            if data_parallel:
                sums = _merge_source_sums_across_devices(sums)
            return sums

        if data_parallel:
            chunk_sums = parallel_jax.jit_sharded(
                chunk_sums_impl,
                in_specs=(
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    batched_data.partition_spec,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                ),
                out_specs=parallel_jax.SHARE_PARTITION,
                check_vma=True,
            )
            logger.info(
                "Configured held-out LIT data parallelism devices=%d",
                device_count,
            )
        else:
            chunk_sums = jax.jit(chunk_sums_impl)

        cache[key] = (response_apply, ground_logpsi, chunk_sums)
        return chunk_sums

    def _lit_stats_chunked(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
        return_block_sums: bool = False,
    ):
        chunk_size = self._lit_eval_batch_size()
        self._validate_data_parallel_batch(
            batched_data,
            purpose="held-out evaluation pool",
        )
        chunk_sums = self._lit_chunk_sums_kernel(
            response_apply,
            ground_logpsi,
            batched_data,
            axis=axis,
        )
        source_center_array = jnp.asarray(source_center)
        ground_energy_array = jnp.asarray(ground_energy)
        omega_array = jnp.asarray(omega)
        eta_array = jnp.asarray(self.lit_config.eta)
        source_floor_array = jnp.asarray(self.lit_config.source.floor)
        kernel_response_params = response_params
        kernel_ground_params = ground_params
        kernel_source_center = source_center_array
        kernel_ground_energy = ground_energy_array
        kernel_omega = omega_array
        kernel_eta = eta_array
        kernel_source_floor = source_floor_array
        if self._lit_data_parallel_enabled():
            kernel_response_params = _replicate_across_local_devices(response_params)
            kernel_ground_params = _replicate_across_local_devices(ground_params)
            kernel_source_center = _replicate_across_local_devices(source_center_array)
            kernel_ground_energy = _replicate_across_local_devices(ground_energy_array)
            kernel_omega = _replicate_across_local_devices(omega_array)
            kernel_eta = _replicate_across_local_devices(eta_array)
            kernel_source_floor = _replicate_across_local_devices(source_floor_array)

        total_sums = None
        block_sums = []
        for chunk in _batched_data_chunks(batched_data, chunk_size):
            self._validate_data_parallel_batch(chunk, purpose="evaluation")
            kernel_chunk = (
                _shard_batched_data_across_local_devices(chunk)
                if self._lit_data_parallel_enabled()
                else chunk
            )
            sums = chunk_sums(
                kernel_response_params,
                kernel_ground_params,
                kernel_chunk,
                kernel_source_center,
                kernel_ground_energy,
                kernel_omega,
                kernel_eta,
                kernel_source_floor,
            )
            if return_block_sums:
                block_sums.append(sums)
            total_sums = (
                sums if total_sums is None else _add_source_sums(total_sums, sums)
            )
        if total_sums is None:
            msg = "Cannot evaluate LIT stats with an empty source pool."
            raise ValueError(msg)
        stats = _lit_stats_from_source_sums(
            total_sums,
            jnp.asarray(source_norm),
            omega_array,
            eta_array,
        )
        if return_block_sums:
            return stats, tuple(block_sums)
        return stats

    def _log_response_pool_capacity(
        self,
        response_params,
        train_pool: BatchedData,
        eval_pool: BatchedData,
        *,
        axis: int,
    ) -> None:
        parameter_count = int(ravel_pytree(response_params)[0].size)
        train_walkers = int(train_pool.batch_size)
        eval_walkers = int(eval_pool.batch_size)
        train_batch = self._lit_train_update_batch_size()
        eval_batch = self._lit_eval_batch_size()
        device_count = (
            jax.local_device_count() if self._lit_data_parallel_enabled() else 1
        )
        denominator = max(parameter_count, 1)
        logger.info(
            "axis=%s response_parameter_count=%d raw_train_walkers=%d "
            "raw_eval_walkers=%d train_walkers_per_parameter=%.3f "
            "eval_walkers_per_parameter=%.3f global_train_batch=%d "
            "per_device_train_batch=%d global_eval_batch=%d "
            "per_device_eval_batch=%d devices=%d",
            _AXIS_NAMES[axis],
            parameter_count,
            train_walkers,
            eval_walkers,
            train_walkers / denominator,
            eval_walkers / denominator,
            train_batch,
            train_batch // device_count,
            eval_batch,
            eval_batch // device_count,
            device_count,
        )
        if train_walkers < parameter_count:
            logger.warning(
                "axis=%s fixed response train pool has fewer raw walkers (%d) "
                "than response parameters (%d); held-out overfitting risk is high",
                _AXIS_NAMES[axis],
                train_walkers,
                parameter_count,
            )

    def _lit_train_update_batch_size(self) -> int:
        return self._lit_effective_batch_size(
            global_size=self.lit_config.parallel.train_batch_size,
            per_device_size=(self.lit_config.parallel.train_batch_size_per_device),
        )

    def _lit_eval_batch_size(self) -> int:
        return self._lit_effective_batch_size(
            global_size=self.lit_config.parallel.eval_batch_size,
            per_device_size=self.lit_config.parallel.eval_batch_size_per_device,
        )

    def _lit_effective_batch_size(
        self,
        *,
        global_size: int,
        per_device_size: int,
    ) -> int:
        """Resolve a configured global or per-device walker count.

        Returns:
            The global batch size used by the fixed-pool kernels.

        Raises:
            ValueError: If both global and per-device sizes are positive.
        """
        configured_global = int(global_size)
        configured_per_device = int(per_device_size)
        if configured_global > 0 and configured_per_device > 0:
            raise ValueError("Global and per-device NQS batch sizes are exclusive.")
        if configured_global > 0:
            return configured_global
        if configured_per_device > 0:
            device_count = (
                jax.local_device_count() if self._lit_data_parallel_enabled() else 1
            )
            return configured_per_device * device_count
        return max(1, int(self.config.batch_size))

    def _training_shuffle_seed(
        self,
        *,
        axis: int,
        stage: str,
        omega: float | None = None,
    ) -> int:
        configured_seed = getattr(getattr(self, "config", None), "seed", None)
        base_seed = (
            int(configured_seed)
            if configured_seed is not None
            else int(getattr(self, "_run_seed", 0))
        )
        frequency = "none" if omega is None else float(omega).hex()
        payload = f"{base_seed}:{int(axis)}:{stage}:{frequency}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def _lit_data_parallel_enabled(self) -> bool:
        return self._lit_data_parallel_mode() == "local_devices"

    def _lit_data_parallel_mode(self) -> str:
        # YAML 1.1 parsers commonly decode an unquoted ``off`` as ``False``.
        # Preserve that established spelling without accepting ``True`` as a
        # second, underspecified parallel mode.
        configured = self.lit_config.parallel.mode
        if configured is False:
            return "off"
        return str(configured).lower()

    def _validate_data_parallel_batch_size(
        self,
        batch_size: int,
        *,
        purpose: str,
    ) -> None:
        if not self._lit_data_parallel_enabled():
            return
        device_count = jax.local_device_count()
        if batch_size < device_count or batch_size % device_count != 0:
            msg = (
                f"Data-parallel LIT {purpose} batch size {batch_size} must "
                f"be a positive multiple of the {device_count} local devices."
            )
            raise ValueError(msg)

    def _validate_data_parallel_batch(self, batched_data, *, purpose: str) -> None:
        self._validate_data_parallel_batch_size(
            int(batched_data.batch_size),
            purpose=purpose,
        )

    def _validate_source_pool_chunks(
        self,
        train_pool,
        eval_pool,
    ) -> None:
        """Validate complete, fixed-size train/evaluation chunk partitions.

        Raises:
            ValueError: If a pool is empty, has a partial chunk, or cannot shard.
        """
        pool_specs = (
            ("training", train_pool, self._lit_train_update_batch_size()),
            ("held-out evaluation", eval_pool, self._lit_eval_batch_size()),
        )
        for purpose, pool, requested_size in pool_specs:
            pool_size = int(pool.batch_size)
            chunk_size = min(int(requested_size), pool_size)
            if chunk_size < 1:
                raise ValueError(f"LIT {purpose} source pool is empty.")
            if chunk_size < pool_size and pool_size % chunk_size != 0:
                raise ValueError(
                    f"LIT {purpose} source pool size {pool_size} must be "
                    f"divisible by its effective chunk size {chunk_size}."
                )
            self._validate_data_parallel_batch_size(
                chunk_size,
                purpose=f"{purpose} chunk",
            )

    def _data_parallel_source_update_kernel(self, source_update, update_batch):
        self._validate_data_parallel_batch(update_batch, purpose="training")
        device_count = jax.local_device_count()
        logger.info(
            "Compiling single-frequency LIT data parallelism devices=%d "
            "global_train_batch=%d local_train_batch=%d",
            device_count,
            int(update_batch.batch_size),
            int(update_batch.batch_size) // device_count,
        )
        return parallel_jax.jit_sharded(
            source_update,
            in_specs=(
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
                update_batch.partition_spec,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
            ),
            out_specs=parallel_jax.SHARE_PARTITION,
            check_vma=True,
        )

    def _init_lit_update_carry(
        self,
        rng,
        response_params,
    ) -> _LITUpdateCarry:
        flat_params, _ = ravel_pytree(response_params)
        return _LITUpdateCarry(
            spring=_SpringState(previous_direction=jnp.zeros_like(flat_params)),
            rng=rng,
        )

    def _log_lit_summary(self, output_path: str, fidelity: np.ndarray) -> None:
        logger.info("Wrote LIT spectrum to %s", output_path)
        logger.info(
            "LIT fidelity range: min=%.6f max=%.6f",
            float(np.min(fidelity)),
            float(np.max(fidelity)),
        )
