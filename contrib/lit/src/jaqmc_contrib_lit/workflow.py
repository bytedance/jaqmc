# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular dipole LIT workflow."""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import data_init
from jaqmc.app.molecule.workflow import configure_system
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ConfigManager
from jaqmc.workflow.base import Workflow
from jaqmc_contrib_lit.common import (
    _AXIS_NAMES,
    _axis_indices,
    _lit_omega_grid,
    _optional_float,
    _save_npz,
    _three_component_override,
)
from jaqmc_contrib_lit.config import LITConfig
from jaqmc_contrib_lit.continuation_policy import (
    _continuation_checkpoint_digests,
    _source_pool_target_digest,
    _tree_content_digest,
)
from jaqmc_contrib_lit.inversion import lit_block_statistics
from jaqmc_contrib_lit.optimization import _lit_error_monitor, _regularized_loss
from jaqmc_contrib_lit.pool import _signed_lit_jackknife_pseudovalues
from jaqmc_contrib_lit.scan import ScanMixin
from jaqmc_contrib_lit.sector import (
    _is_atom_hard_parity_sector,
    _resolve_atomic_parity_sector,
    _response_symmetry_policy,
)
from jaqmc_contrib_lit.source import SourceStageMixin
from jaqmc_contrib_lit.state import (
    _ContinuationRecord,
)
from jaqmc_contrib_lit.training import TrainingMixin

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


class LITWorkflow(SourceStageMixin, TrainingMixin, ScanMixin, Workflow):
    """Compute a molecular dipole response spectrum with LIT."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        self.lit_config = cfg.get("lit", LITConfig)
        self.system_config, self.wf = configure_system(cfg)
        self.sampler = cfg.get("sampler", MCMCSampler)
        self._lit_chunk_sums_kernel_cache: dict[
            tuple[int, int, int],
            tuple[Any, Any, Any],
        ] = {}
        self._source_distillation_eval_kernel_cache: dict[tuple[Any, ...], Any] = {}
        self._source_sample_step_kernel_cache: dict[int, Any] = {}
        self._validate_config()

    def run(self) -> None:
        self._run_serial_scan()

    def _run_serial_scan(self) -> None:
        axes = _axis_indices(self.lit_config.axes)
        omega = self._omega_grid()
        seed = self.config.seed if self.config.seed is not None else int(time.time())
        self._run_seed = int(seed)
        rng = jax.random.PRNGKey(seed)
        rng, data_rng, ground_rng, response_rng, sample_rng = jax.random.split(rng, 5)
        batched_data = data_init(self.system_config, self.config.batch_size, data_rng)
        # Resolve the deliberately narrow atom/C1 policy from physical fixed
        # fields before checkpoint loading, sampling, or compilation.  The
        # shape-only example replaces fixed values and must never classify the
        # geometry.
        source_sector = self._configured_source_sector(batched_data.data)
        shape_example = batched_data.unbatched_example()

        checkpoint_step, ground_params, ground_logpsi = self._resolve_lit_ground_state(
            shape_example, ground_rng
        )
        source_pool_target_sha256 = _source_pool_target_digest(
            ground_params,
            batched_data.data,
        )
        ground_sample_plan = SamplePlan(ground_logpsi, {"electrons": self.sampler})
        sampler_state = ground_sample_plan.init(batched_data, sample_rng)
        for _ in range(self.lit_config.ground.burn_in):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = ground_sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )

        ground_energy, batched_data, sampler_state, rng = self._resolve_ground_energy(
            ground_logpsi,
            ground_params,
            batched_data,
            sampler_state,
            ground_sample_plan,
            rng,
        )
        logger.info("Using LIT ground energy %.10f Ha", ground_energy)
        parity_resolution = self._resolve_atomic_parity(
            ground_logpsi,
            ground_params,
            batched_data,
            source_sector,
        )
        source_sector = _resolve_atomic_parity_sector(
            source_sector,
            parity_resolution.response_parity,
        )

        signed_lit = np.zeros((len(axes), len(omega)), dtype=np.float64)
        broadened = np.zeros_like(signed_lit)
        fidelity = np.zeros_like(signed_lit)
        reverse_kl = np.zeros_like(signed_lit)
        residual_norm = np.zeros_like(signed_lit)
        equation_relative_residual = np.zeros_like(signed_lit)
        action_norm = np.zeros_like(signed_lit)
        source_norm = np.zeros_like(signed_lit)
        error_bound_monitor = np.zeros_like(signed_lit)
        error_d = np.zeros_like(signed_lit)
        error_d_correction = np.zeros_like(signed_lit)
        error_d_shifted = np.zeros_like(signed_lit)
        error_d_valid = np.zeros_like(signed_lit, dtype=np.bool_)
        reweight_ess = np.zeros_like(signed_lit)
        reweight_ess_fraction = np.zeros_like(signed_lit)
        reweight_max_fraction = np.zeros_like(signed_lit)
        invalid_sample_fraction = np.zeros_like(signed_lit)
        selected_iteration = np.zeros_like(signed_lit, dtype=np.int64)
        normalization = np.zeros((len(axes), len(omega)), dtype=np.complex128)
        correction_overlap = np.zeros_like(normalization)
        source_centers = np.zeros(len(axes), dtype=np.float64)
        axis_source_norm = np.zeros(len(axes), dtype=np.float64)
        atomic_source_parity_loss = np.full(len(axes), np.nan, dtype=np.float64)
        eval_pool_sha256 = np.empty(len(axes), dtype="<U64")
        axis_jackknife_blocks: list[np.ndarray] = []
        warm_start_selected_iteration = np.zeros(len(axes), dtype=np.int64)
        continuation_axis: list[int] = []
        continuation_omega: list[float] = []
        continuation_optimized: list[bool] = []
        continuation_selected_iteration: list[int] = []
        continuation_fidelity: list[float] = []
        continuation_reverse_kl: list[float] = []
        continuation_invalid_sample_fraction: list[float] = []
        continuation_inherited_fidelity: list[float] = []
        continuation_step: list[float] = []
        continuation_bisections: list[int] = []
        continuation_probe_accepted: list[bool] = []
        continuation_min_step_override: list[bool] = []

        logger.info(
            "LIT response_policy=%s source_sector=%s order=%d",
            _response_symmetry_policy(
                source_sector,
                parity_resolution.response_parity,
            ),
            source_sector.label,
            source_sector.order,
        )

        # Estimate all three dipole centers on one ground-state chain.  The
        # atomic inversion sector is retained only to project the affine,
        # origin-dependent center before norms are finalized; both supported
        # response policies themselves use one scalar axis at a time.
        (
            vector_source_centers,
            vector_source_norms,
            batched_data,
            sampler_state,
            rng,
        ) = self._estimate_vector_source_stats(
            ground_params,
            batched_data,
            sampler_state,
            ground_sample_plan,
            rng,
            source_sector=source_sector,
        )

        for axis_pos, axis in enumerate(axes):
            source_center = float(vector_source_centers[axis])
            axis_phi_norm = float(vector_source_norms[axis])
            source_centers[axis_pos] = source_center
            axis_source_norm[axis_pos] = axis_phi_norm
            logger.info(
                "axis=%s source_center=%.8e source_norm=%.8e",
                _AXIS_NAMES[axis],
                source_center,
                axis_phi_norm,
            )

            rng, response_rng = jax.random.split(rng)
            response_apply, response_params = self._make_response_ansatz(
                shape_example,
                response_rng,
                ground_params,
                source_sector=source_sector,
                response_parity=parity_resolution.response_parity,
            )
            loaded_pools = self._try_load_source_pools(
                batched_data,
                axis=axis,
                source_center=source_center,
                target_sha256=source_pool_target_sha256,
            )
            if loaded_pools is None:
                source_sample_plan, source_state, axis_batched_data, rng = (
                    self._prepare_source_sampler(
                        self.sampler,
                        batched_data,
                        ground_params,
                        ground_logpsi,
                        rng,
                        axis=axis,
                        source_center=source_center,
                    )
                )
                train_pool, axis_batched_data, source_state, rng = (
                    self._load_or_collect_source_pool(
                        source_sample_plan,
                        ground_params,
                        axis_batched_data,
                        source_state,
                        rng,
                        axis=axis,
                        source_center=source_center,
                        target_sha256=source_pool_target_sha256,
                        split="train",
                        batches=self.lit_config.source.train_pool_batches,
                    )
                )
                eval_pool, axis_batched_data, source_state, rng = (
                    self._load_or_collect_source_pool(
                        source_sample_plan,
                        ground_params,
                        axis_batched_data,
                        source_state,
                        rng,
                        axis=axis,
                        source_center=source_center,
                        target_sha256=source_pool_target_sha256,
                        split="eval",
                        batches=self.lit_config.source.eval_pool_batches,
                    )
                )
            else:
                train_pool, eval_pool = loaded_pools
                axis_batched_data = batched_data
            self._validate_source_pool_chunks(train_pool, eval_pool)
            eval_pool_sha256[axis_pos] = _tree_content_digest(eval_pool)
            self._log_response_pool_capacity(
                response_params,
                train_pool,
                eval_pool,
                axis=axis,
            )
            atomic_source_parity_loss[axis_pos] = self._validate_atomic_source_parity(
                ground_logpsi,
                ground_params,
                eval_pool,
                source_sector,
                vector_source_centers,
                axis=axis,
                response_parity=parity_resolution.response_parity,
            )
            update_step = self._make_lit_update_step(
                response_apply,
                ground_params,
                ground_logpsi,
                ground_energy,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
            )
            state_fingerprint, full_config_digest = _continuation_checkpoint_digests(
                self.lit_config,
                response_params=response_params,
                ground_params=ground_params,
                train_pool=train_pool,
                eval_pool=eval_pool,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
                ground_checkpoint_step=checkpoint_step,
                response_parity=parity_resolution.response_parity,
                target_omega=float(omega[0]),
                spectrum_omega=omega,
            )
            resume_state = self._restore_lit_continuation_checkpoint(
                response_params,
                rng,
                eval_pool,
                response_apply=response_apply,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
                ground_checkpoint_step=checkpoint_step,
                response_parity=parity_resolution.response_parity,
                target_omega=float(omega[0]),
                state_fingerprint=state_fingerprint,
                full_config_digest=full_config_digest,
            )
            if resume_state is None:
                response_params, rng = self._distill_response_from_source(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    train_pool,
                    eval_pool,
                    rng,
                    axis=axis,
                    source_center=source_center,
                )
                (
                    response_params,
                    continuation_start_stats,
                    warm_start_selected_iteration[axis_pos],
                    rng,
                ) = self._warm_start_axis(
                    update_step,
                    response_params,
                    train_pool,
                    eval_pool,
                    rng,
                    response_apply=response_apply,
                    ground_logpsi=ground_logpsi,
                    ground_params=ground_params,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                )
                resume_omega = None
                existing_records: tuple[_ContinuationRecord, ...] = ()
            else:
                response_params = resume_state.response_params
                rng = resume_state.rng
                continuation_start_stats = resume_state.current_stats
                warm_start_selected_iteration[axis_pos] = (
                    resume_state.warm_start_selected_iteration
                )
                resume_omega = resume_state.current_omega
                existing_records = resume_state.records
            checkpoint_callback = partial(
                self._save_lit_continuation_checkpoint,
                axis=axis,
                target_omega=float(omega[0]),
                ground_checkpoint_step=checkpoint_step,
                ground_energy=ground_energy,
                source_center=source_center,
                source_norm=axis_phi_norm,
                response_parity=parity_resolution.response_parity,
                state_fingerprint=state_fingerprint,
                full_config_digest=full_config_digest,
                warm_start_selected_iteration=int(
                    warm_start_selected_iteration[axis_pos]
                ),
            )
            response_params, _, bridge_records, rng = self._continue_lit_to_spectrum(
                update_step,
                response_params,
                continuation_start_stats,
                train_pool,
                eval_pool,
                rng,
                response_apply=response_apply,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
                target_omega=float(omega[0]),
                spectrum_omega=omega,
                resume_omega=resume_omega,
                existing_records=existing_records,
                checkpoint_callback=checkpoint_callback,
            )
            for record in bridge_records:
                host_bridge_stats = jax.device_get(record.stats)
                continuation_axis.append(axis)
                continuation_omega.append(record.omega)
                continuation_optimized.append(record.optimized)
                continuation_selected_iteration.append(record.selected_iteration)
                continuation_fidelity.append(float(host_bridge_stats.fidelity))
                continuation_reverse_kl.append(float(host_bridge_stats.reverse_kl))
                continuation_invalid_sample_fraction.append(
                    float(host_bridge_stats.invalid_sample_fraction)
                )
                continuation_inherited_fidelity.append(record.inherited_fidelity)
                continuation_step.append(record.step)
                continuation_bisections.append(record.bisections)
                continuation_probe_accepted.append(record.probe_accepted)
                continuation_min_step_override.append(record.min_step_override)
            logger.info(
                "axis=%s frequency_continuation=serial bridge_points=%d "
                "spectrum_points=%d",
                _AXIS_NAMES[axis],
                sum(record.optimized for record in bridge_records),
                len(omega),
            )
            omega_jackknife_blocks: list[np.ndarray] = []
            for omega_pos, omega_value in enumerate(omega):
                (
                    response_params,
                    _,
                    selected_iteration[axis_pos, omega_pos],
                    rng,
                ) = self._optimize_lit_frequency(
                    update_step,
                    response_params,
                    train_pool,
                    eval_pool,
                    rng,
                    response_apply=response_apply,
                    ground_logpsi=ground_logpsi,
                    ground_params=ground_params,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                    omega=float(omega_value),
                    iterations=self.lit_config.solver.iterations,
                    stage="spectrum",
                )
                stats, block_sums = self._lit_stats_chunked(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    eval_pool,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                    omega=jnp.asarray(float(omega_value)),
                    return_block_sums=True,
                )
                stats = stats._replace(
                    loss=_regularized_loss(
                        stats,
                        self.lit_config.solver.reverse_kl_weight,
                    )
                )
                host_stats = jax.device_get(stats)
                omega_jackknife_blocks.append(
                    _signed_lit_jackknife_pseudovalues(
                        stats,
                        block_sums,
                        source_norm=axis_phi_norm,
                        omega=float(omega_value),
                        eta=float(self.lit_config.eta),
                    )
                )
                signed_lit[axis_pos, omega_pos] = float(host_stats.signed_lit)
                broadened[axis_pos, omega_pos] = float(host_stats.broadened)
                fidelity[axis_pos, omega_pos] = float(host_stats.fidelity)
                reverse_kl[axis_pos, omega_pos] = float(host_stats.reverse_kl)
                residual_norm[axis_pos, omega_pos] = float(host_stats.residual_norm)
                equation_relative_residual[axis_pos, omega_pos] = float(
                    host_stats.equation_relative_residual
                )
                action_norm[axis_pos, omega_pos] = float(host_stats.action_norm)
                source_norm[axis_pos, omega_pos] = float(host_stats.source_norm)
                error_bound_monitor[axis_pos, omega_pos] = _lit_error_monitor(
                    fidelity=float(host_stats.fidelity),
                    source_norm=float(host_stats.source_norm),
                    normalization=complex(host_stats.normalization),
                    eta=float(self.lit_config.eta),
                    error_d=float(host_stats.error_d),
                    error_d_valid=bool(host_stats.error_d_valid),
                )
                error_d[axis_pos, omega_pos] = float(host_stats.error_d)
                error_d_correction[axis_pos, omega_pos] = float(
                    host_stats.error_d_correction
                )
                error_d_shifted[axis_pos, omega_pos] = float(host_stats.error_d_shifted)
                error_d_valid[axis_pos, omega_pos] = bool(host_stats.error_d_valid)
                reweight_ess[axis_pos, omega_pos] = float(host_stats.reweight_ess)
                reweight_ess_fraction[axis_pos, omega_pos] = float(
                    host_stats.reweight_ess_fraction
                )
                reweight_max_fraction[axis_pos, omega_pos] = float(
                    host_stats.reweight_max_fraction
                )
                invalid_sample_fraction[axis_pos, omega_pos] = float(
                    host_stats.invalid_sample_fraction
                )
                normalization[axis_pos, omega_pos] = complex(host_stats.normalization)
                correction_overlap[axis_pos, omega_pos] = complex(
                    host_stats.correction_overlap
                )
            axis_jackknife_blocks.append(np.stack(omega_jackknife_blocks, axis=0))

        signed_lit_jackknife_blocks = np.stack(axis_jackknife_blocks, axis=0)
        uncertainty_output: dict[str, Any] = {}
        if signed_lit_jackknife_blocks.shape[-1] >= 2:
            block_statistics = lit_block_statistics(signed_lit_jackknife_blocks)
            uncertainty_output = {
                "signed_lit_jackknife_blocks": signed_lit_jackknife_blocks,
                "signed_lit_jackknife_block_count": np.asarray(
                    block_statistics.block_count,
                    dtype=np.int64,
                ),
                "signed_lit_covariance": block_statistics.covariance,
                "signed_lit_standard_error": block_statistics.standard_error,
            }
        total_broadened = np.sum(broadened, axis=0)
        output_path = self.save_path / self.lit_config.output_filename
        _save_npz(
            output_path,
            backend="lit",
            omega=omega,
            eta=self.lit_config.eta,
            axes=self.lit_config.axes,
            axis_indices=np.asarray(axes, dtype=np.int64),
            signed_lit=signed_lit,
            broadened=broadened,
            total_broadened=total_broadened,
            fidelity=fidelity,
            reverse_kl=reverse_kl,
            residual_norm=residual_norm,
            equation_relative_residual=equation_relative_residual,
            action_norm=action_norm,
            source_norm=source_norm,
            error_bound_monitor=error_bound_monitor,
            error_d=error_d,
            error_d_correction=error_d_correction,
            error_d_shifted=error_d_shifted,
            error_d_valid=error_d_valid,
            reweight_ess=reweight_ess,
            reweight_ess_fraction=reweight_ess_fraction,
            reweight_max_fraction=reweight_max_fraction,
            invalid_sample_fraction=invalid_sample_fraction,
            selected_iteration=selected_iteration,
            normalization=normalization,
            correction_overlap=correction_overlap,
            ground_energy=ground_energy,
            ground_checkpoint_step=checkpoint_step,
            lit_train_pool_batches=self.lit_config.source.train_pool_batches,
            lit_eval_pool_batches=self.lit_config.source.eval_pool_batches,
            lit_pool_stride=self.lit_config.source.pool_stride,
            lit_reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
            lit_spring_epsilon=self.lit_config.solver.spring_epsilon,
            lit_spring_decay=self.lit_config.solver.spring_decay,
            lit_spring_damping_floor=self.lit_config.solver.spring_damping_floor,
            lit_source_distillation_iterations=(
                self.lit_config.source.distillation_iterations
            ),
            lit_parity_eval_batch_size=self.lit_config.ansatz.parity_eval_batch_size,
            lit_sector_tolerance=self.lit_config.ansatz.sector_tolerance,
            lit_atomic_source_parity_max_loss=(
                self.lit_config.ansatz.atomic_source_parity_max_loss
            ),
            lit_atomic_ground_parity_max_loss=(
                self.lit_config.ansatz.atomic_ground_parity_max_loss
            ),
            source_sector_label=source_sector.label,
            source_sector_order=source_sector.order,
            response_symmetry_policy=_response_symmetry_policy(
                source_sector,
                parity_resolution.response_parity,
            ),
            response_hard_parity=bool(_is_atom_hard_parity_sector(source_sector)),
            atomic_ground_parity=parity_resolution.ground_parity,
            response_parity=parity_resolution.response_parity,
            atomic_ground_even_parity_loss=parity_resolution.even_loss,
            atomic_ground_odd_parity_loss=parity_resolution.odd_loss,
            atomic_ground_selected_parity_loss=parity_resolution.selected_loss,
            response_symmetry_center=np.asarray(
                source_sector.center,
                dtype=np.float64,
            ),
            vector_source_centers=vector_source_centers,
            vector_source_norms=vector_source_norms,
            atomic_source_parity_loss=atomic_source_parity_loss,
            lit_selection_interval=self.lit_config.solver.selection_interval,
            lit_warm_start_omega=_optional_float(
                self.lit_config.solver.warm_start_omega
            ),
            lit_warm_start_iterations=self.lit_config.solver.warm_start_iterations,
            warm_start_selected_iteration=warm_start_selected_iteration,
            lit_fidelity_plateau_start_iteration=(
                self.lit_config.solver.plateau_start_iteration
            ),
            lit_fidelity_plateau_patience_iterations=(
                self.lit_config.solver.plateau_patience_iterations
            ),
            lit_fidelity_plateau_min_delta=(self.lit_config.solver.plateau_min_delta),
            continuation_axis=np.asarray(continuation_axis, dtype=np.int64),
            continuation_omega=np.asarray(continuation_omega, dtype=np.float64),
            continuation_optimized=np.asarray(
                continuation_optimized,
                dtype=np.bool_,
            ),
            continuation_selected_iteration=np.asarray(
                continuation_selected_iteration,
                dtype=np.int64,
            ),
            continuation_fidelity=np.asarray(
                continuation_fidelity,
                dtype=np.float64,
            ),
            continuation_reverse_kl=np.asarray(
                continuation_reverse_kl,
                dtype=np.float64,
            ),
            continuation_invalid_sample_fraction=np.asarray(
                continuation_invalid_sample_fraction,
                dtype=np.float64,
            ),
            continuation_inherited_fidelity=np.asarray(
                continuation_inherited_fidelity,
                dtype=np.float64,
            ),
            continuation_step=np.asarray(continuation_step, dtype=np.float64),
            continuation_bisections=np.asarray(
                continuation_bisections,
                dtype=np.int64,
            ),
            continuation_probe_accepted=np.asarray(
                continuation_probe_accepted,
                dtype=np.bool_,
            ),
            continuation_min_step_override=np.asarray(
                continuation_min_step_override,
                dtype=np.bool_,
            ),
            lit_continuation_iterations=self.lit_config.continuation.iterations,
            lit_continuation_step_fraction=(self.lit_config.continuation.step_fraction),
            lit_continuation_step_growth_factor=(
                self.lit_config.continuation.step_growth_factor
            ),
            lit_continuation_fidelity_retention=(
                self.lit_config.continuation.fidelity_retention
            ),
            lit_stage_reweight_ess_fraction_min=(
                self.lit_config.continuation.ess_fraction_minimum
            ),
            lit_continuation_allow_min_step_override=bool(
                self.lit_config.continuation.allow_minimum_step_recovery
            ),
            lit_continuation_min_step=_optional_float(
                self.lit_config.continuation.minimum_step
            ),
            lit_continuation_max_points=self.lit_config.continuation.maximum_points,
            lit_continuation_restore_path=(self.lit_config.continuation.restore_path),
            source_centers=source_centers,
            axis_source_norm=axis_source_norm,
            eval_pool_sha256=eval_pool_sha256,
            **uncertainty_output,
        )
        self._log_lit_summary(str(output_path), fidelity)

    def _omega_grid(self) -> np.ndarray:
        return _lit_omega_grid(self.lit_config)

    def _validate_config(self) -> None:
        self._validate_hamiltonian_config()
        omega = _lit_omega_grid(self.lit_config)
        if not np.isfinite(self.lit_config.eta) or self.lit_config.eta <= 0.0:
            msg = "lit.eta must be finite and positive."
            raise ValueError(msg)
        self._validate_serial_scan_config(omega)
        self._validate_chunk_config()
        self._validate_data_parallel_config()
        self._validate_lit_stabilizer_config()
        self._validate_source_sector_config()
        self._validate_lit_iteration_config()
        self._validate_continuation_config()

    def _validate_hamiltonian_config(self) -> None:
        pseudopotentials = sorted(
            {
                (atom.symbol, atom.pp)
                for atom in self.system_config.atoms
                if atom.pp is not None
            }
        )
        if not pseudopotentials:
            return
        configured = ", ".join(
            f"{symbol}={pseudopotential}"
            for symbol, pseudopotential in pseudopotentials
        )
        raise ValueError(
            "LIT currently supports only all-electron molecular Hamiltonians; "
            "ECP and PH pseudopotentials are not implemented in the LIT local "
            f"action. Set system.pp to null; got {configured}."
        )

    def _validate_serial_scan_config(self, omega: np.ndarray) -> None:
        warm_omega = self.lit_config.solver.warm_start_omega
        if warm_omega is not None:
            if not np.isfinite(warm_omega):
                msg = "lit.solver.warm_start_omega must be finite or null."
                raise ValueError(msg)
            if warm_omega >= float(omega[0]):
                msg = (
                    "lit.solver.warm_start_omega must be below the first spectrum "
                    "frequency for increasing serial continuation."
                )
                raise ValueError(msg)

    def _validate_lit_stabilizer_config(self) -> None:
        if (
            not np.isfinite(self.lit_config.solver.reverse_kl_weight)
            or self.lit_config.solver.reverse_kl_weight < 0.0
        ):
            msg = "lit.solver.reverse_kl_weight must be nonnegative."
            raise ValueError(msg)
        if (
            not np.isfinite(self.lit_config.solver.spring_epsilon)
            or self.lit_config.solver.spring_epsilon <= 0.0
        ):
            msg = "lit.solver.spring_epsilon must be positive."
            raise ValueError(msg)
        if not 0.0 <= self.lit_config.solver.spring_decay < 1.0:
            msg = "lit.solver.spring_decay must satisfy 0 <= value < 1."
            raise ValueError(msg)
        if (
            not np.isfinite(self.lit_config.solver.spring_damping_floor)
            or self.lit_config.solver.spring_damping_floor <= 0.0
        ):
            msg = "lit.solver.spring_damping_floor must be positive."
            raise ValueError(msg)
        if (
            not np.isfinite(self.lit_config.solver.learning_rate)
            or self.lit_config.solver.learning_rate <= 0.0
        ):
            msg = "lit.solver.learning_rate must be finite and positive."
            raise ValueError(msg)
        max_norm = self.lit_config.solver.sr_max_norm
        if max_norm is not None and (
            not np.isfinite(max_norm) or float(max_norm) <= 0.0
        ):
            msg = "lit.solver.sr_max_norm must be positive or null."
            raise ValueError(msg)

    def _validate_lit_iteration_config(self) -> None:
        if self.lit_config.solver.selection_interval < 1:
            msg = "lit.solver.selection_interval must be positive."
            raise ValueError(msg)
        if self.lit_config.solver.warm_start_iterations < 0:
            msg = "lit.solver.warm_start_iterations must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.solver.iterations < 1:
            msg = "lit.solver.iterations must be positive."
            raise ValueError(msg)
        distillation_iterations = self.lit_config.source.distillation_iterations
        if (
            isinstance(distillation_iterations, (bool, np.bool_))
            or not isinstance(distillation_iterations, (int, np.integer))
            or int(distillation_iterations) < 1
        ):
            msg = "lit.source.distillation_iterations must be a positive integer."
            raise ValueError(msg)
        plateau_integers = (
            (
                "lit.solver.plateau_start_iteration",
                self.lit_config.solver.plateau_start_iteration,
            ),
            (
                "lit.solver.plateau_patience_iterations",
                self.lit_config.solver.plateau_patience_iterations,
            ),
        )
        for name, value in plateau_integers:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer.")
        plateau_delta = self.lit_config.solver.plateau_min_delta
        if not np.isfinite(plateau_delta) or not 0.0 <= float(plateau_delta) <= 1.0:
            raise ValueError(
                "lit.solver.plateau_min_delta must be finite and between 0 and 1."
            )

    def _validate_source_sector_config(self) -> None:
        _three_component_override(
            self.lit_config.source.center_override,
            name="lit.source.center_override",
        )
        _three_component_override(
            self.lit_config.source.norm_override,
            name="lit.source.norm_override",
            positive=True,
        )
        if self.lit_config.ansatz.parity_eval_batch_size < 1:
            msg = "lit.ansatz.parity_eval_batch_size must be positive."
            raise ValueError(msg)
        tolerance = self.lit_config.ansatz.sector_tolerance
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            msg = "lit.ansatz.sector_tolerance must be positive."
            raise ValueError(msg)
        source_parity_maximum = self.lit_config.ansatz.atomic_source_parity_max_loss
        if (
            not np.isfinite(source_parity_maximum)
            or not 0.0 < source_parity_maximum < 1.0
        ):
            msg = (
                "lit.ansatz.atomic_source_parity_max_loss must be finite and "
                "strictly between 0 and 1."
            )
            raise ValueError(msg)
        self._validate_atomic_parity_config()

    def _validate_atomic_parity_config(self) -> None:
        """Validate the mandatory atomic checkpoint parity admission guard.

        Raises:
            ValueError: If the threshold is non-finite or outside ``(0, 1)``.
        """
        parity_maximum = self.lit_config.ansatz.atomic_ground_parity_max_loss
        if not np.isfinite(parity_maximum) or not 0.0 < parity_maximum < 1.0:
            msg = (
                "lit.ansatz.atomic_ground_parity_max_loss must be finite and "
                "strictly between 0 and 1."
            )
            raise ValueError(msg)

    def _validate_continuation_config(self) -> None:
        continuation_iterations = self.lit_config.continuation.iterations
        if (
            isinstance(continuation_iterations, (bool, np.bool_))
            or not isinstance(continuation_iterations, (int, np.integer))
            or int(continuation_iterations) < 1
        ):
            msg = "lit.continuation.iterations must be a positive integer."
            raise ValueError(msg)
        step_fraction = self.lit_config.continuation.step_fraction
        if not np.isfinite(step_fraction) or step_fraction <= 0.0:
            msg = "lit.continuation.step_fraction must be positive."
            raise ValueError(msg)
        growth = self.lit_config.continuation.step_growth_factor
        if not np.isfinite(growth) or not 1.0 <= float(growth) <= 2.0:
            msg = (
                "lit.continuation.step_growth_factor must be finite and "
                "satisfy 1 <= value <= 2."
            )
            raise ValueError(msg)
        retention = self.lit_config.continuation.fidelity_retention
        if not 0.0 < retention <= 1.0:
            msg = "lit.continuation.fidelity_retention must satisfy 0 < value <= 1."
            raise ValueError(msg)
        stage_ess = self.lit_config.continuation.ess_fraction_minimum
        if not np.isfinite(stage_ess) or not 0.0 <= float(stage_ess) <= 1.0:
            msg = (
                "lit.continuation.ess_fraction_minimum must be finite and "
                "between 0 and 1."
            )
            raise ValueError(msg)
        min_step = self.lit_config.continuation.minimum_step
        if min_step is not None and (
            not np.isfinite(min_step) or float(min_step) <= 0.0
        ):
            msg = "lit.continuation.minimum_step must be positive or null."
            raise ValueError(msg)
        if self.lit_config.continuation.maximum_points < 1:
            msg = "lit.continuation.maximum_points must be positive."
            raise ValueError(msg)

    def _validate_chunk_config(self) -> None:
        pairs = (
            (
                "lit.parallel.train_batch_size",
                self.lit_config.parallel.train_batch_size,
                "lit.parallel.train_batch_size_per_device",
                self.lit_config.parallel.train_batch_size_per_device,
            ),
            (
                "lit.parallel.eval_batch_size",
                self.lit_config.parallel.eval_batch_size,
                "lit.parallel.eval_batch_size_per_device",
                self.lit_config.parallel.eval_batch_size_per_device,
            ),
        )
        for global_name, global_value, local_name, local_value in pairs:
            if int(global_value) < 0:
                raise ValueError(f"{global_name} must be nonnegative.")
            if int(local_value) < 0:
                raise ValueError(f"{local_name} must be nonnegative.")
            if int(global_value) > 0 and int(local_value) > 0:
                raise ValueError(
                    f"{global_name} and {local_name} are mutually exclusive."
                )
        for name, value in (
            (
                "lit.source.train_pool_batches",
                self.lit_config.source.train_pool_batches,
            ),
            ("lit.source.eval_pool_batches", self.lit_config.source.eval_pool_batches),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

    def _validate_data_parallel_config(self) -> None:
        mode = self._lit_data_parallel_mode()
        if mode not in {"off", "local_devices"}:
            msg = (
                "lit.parallel.mode must be 'off' or 'local_devices', got "
                f"{self.lit_config.parallel.mode!r}."
            )
            raise ValueError(msg)
        if jax.process_count() != 1:
            msg = (
                "JaQMC LIT currently requires exactly one JAX process for every "
                "lit.parallel.mode; launch one process controlling the local "
                "devices on the worker."
            )
            raise ValueError(msg)
        if mode == "off":
            return
        # A fully constructed workflow always has the base workflow config.
        # Keep mode-only validation usable by lightweight object.__new__ unit
        # fixtures that deliberately provide only ``lit_config``.
        if not hasattr(self, "config"):
            return
        self._validate_data_parallel_batch_size(
            self._lit_train_update_batch_size(),
            purpose="training",
        )
        self._validate_data_parallel_batch_size(
            self._lit_eval_batch_size(),
            purpose="evaluation",
        )
