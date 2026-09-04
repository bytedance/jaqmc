# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Frequency continuation and recovery for the molecular LIT workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from upath import UPath

from jaqmc_contrib_lit.checkpoint import LITCheckpointManager
from jaqmc_contrib_lit.common import (
    _AXIS_NAMES,
    _CONTINUATION_CHECKPOINT_PREFIX,
    _CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
)
from jaqmc_contrib_lit.continuation_policy import (
    _bisect_continuation_probe,
    _continuation_capacity_diagnostics,
    _continuation_history_step_cap,
    _continuation_min_step,
    _continuation_records_from_json,
    _continuation_records_to_json,
    _empty_lit_stats,
    _latest_continuation_checkpoint_path,
    _physics_continuation_step,
    _read_continuation_checkpoint_metadata,
    _require_continuation_point_capacity,
)
from jaqmc_contrib_lit.optimization import (
    _lit_stage_ess_failure,
    _require_eligible_lit_checkpoint,
    _require_lit_stage_health,
)
from jaqmc_contrib_lit.state import (
    _ContinuationCheckpoint,
    _ContinuationRecord,
    _ContinuationResumeState,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


class ScanMixin:
    def _continuation_checkpoint_save_dir(self, axis: int) -> UPath | None:
        save_path = getattr(self, "save_path", None)
        if save_path is None:
            return None
        return (
            UPath(save_path)
            / "continuation_checkpoints"
            / (f"axis_{_AXIS_NAMES[axis]}")
        )

    def _continuation_checkpoint_restore_path(
        self,
        axis: int,
    ) -> tuple[UPath | None, bool]:
        """Resolve one optional explicit or same-run continuation restore path.

        Returns:
            Resolved path and whether that exact axis checkpoint is required.
            A configured run/checkpoint root is optional per axis so axes not
            reached before an interruption can start fresh.
        """
        configured = str(self.lit_config.continuation.restore_path).strip()
        explicit = bool(configured)
        if not explicit:
            return self._continuation_checkpoint_save_dir(axis), False

        root = UPath(configured)
        if root.suffix == ".npz":
            return root, True
        axis_name = f"axis_{_AXIS_NAMES[axis]}"
        if root.name == axis_name:
            return root, True
        if not root.exists():
            return root / axis_name, True
        nested_root = root / "continuation_checkpoints"
        if nested_root.exists():
            root = nested_root
        return root / axis_name, False

    def _load_lit_continuation_checkpoint(
        self,
        template: _ContinuationCheckpoint,
        *,
        axis: int,
        state_fingerprint: str,
        full_config_digest: str,
    ) -> tuple[UPath, _ContinuationCheckpoint] | None:
        """Load one structurally compatible latest checkpoint, if present.

        Returns:
            Checkpoint path and restored bundle, or ``None`` for a fresh run.

        Raises:
            RuntimeError: If an explicit checkpoint is absent or a discovered
                checkpoint has incompatible metadata or tree structure.
        """
        restore_path, required = self._continuation_checkpoint_restore_path(axis)
        if restore_path is None:
            return None
        checkpoint_path = _latest_continuation_checkpoint_path(restore_path)
        if checkpoint_path is None:
            if required:
                msg = (
                    "No readable continuation checkpoint found for axis="
                    f"{_AXIS_NAMES[axis]} at explicit restore path {restore_path}."
                )
                raise RuntimeError(msg)
            return None

        metadata = _read_continuation_checkpoint_metadata(checkpoint_path)
        if metadata["schema_version"] != _CONTINUATION_CHECKPOINT_SCHEMA_VERSION:
            msg = (
                f"Continuation checkpoint {checkpoint_path} uses schema "
                f"{metadata['schema_version']}, expected "
                f"{_CONTINUATION_CHECKPOINT_SCHEMA_VERSION}."
            )
            raise RuntimeError(msg)
        if metadata["state_fingerprint"] != state_fingerprint:
            msg = (
                f"Continuation checkpoint {checkpoint_path} is incompatible "
                "with the current physical/ansatz state fingerprint "
                f"({metadata['state_fingerprint']} != {state_fingerprint})."
            )
            raise RuntimeError(msg)
        if metadata["full_config_digest"] != full_config_digest:
            logger.warning(
                "Restoring continuation checkpoint %s with a changed non-state "
                "configuration; current gates will be re-applied (%s != %s)",
                checkpoint_path,
                metadata["full_config_digest"],
                full_config_digest,
            )

        save_dir = self._continuation_checkpoint_save_dir(axis)
        if save_dir is None:
            save_dir = checkpoint_path.parent
        manager = LITCheckpointManager(
            save_dir,
            checkpoint_path,
            prefix=_CONTINUATION_CHECKPOINT_PREFIX,
        )
        try:
            initial_step, restored = manager.restore(template)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"Continuation checkpoint {checkpoint_path} has an incompatible tree."
            raise RuntimeError(msg) from exc
        if initial_step != int(restored.accepted_points):
            msg = (
                f"Continuation checkpoint {checkpoint_path} has inconsistent "
                f"step/count metadata ({initial_step} != "
                f"{int(restored.accepted_points)})."
            )
            raise RuntimeError(msg)
        return checkpoint_path, restored

    def _restore_lit_continuation_checkpoint(
        self,
        response_params_template,
        rng_template,
        eval_pool,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        ground_checkpoint_step: int,
        response_parity: int,
        target_omega: float,
        state_fingerprint: str,
        full_config_digest: str,
    ) -> _ContinuationResumeState | None:
        """Restore and revalidate the latest complete bridge checkpoint.

        Returns:
            Validated resume state, or ``None`` when an implicit same-run
            checkpoint does not exist.

        Raises:
            RuntimeError: If an explicit checkpoint is absent or a discovered
                checkpoint is incompatible, malformed, or fails current gates.
        """
        template = _ContinuationCheckpoint(
            schema_version=_CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
            state_fingerprint=state_fingerprint,
            full_config_digest=full_config_digest,
            axis=int(axis),
            target_omega=float(target_omega),
            current_omega=float(
                self.lit_config.solver.warm_start_omega or target_omega
            ),
            accepted_points=0,
            ground_checkpoint_step=int(ground_checkpoint_step),
            ground_energy=float(ground_energy),
            source_center=float(source_center),
            source_norm=float(source_norm),
            response_parity=int(response_parity),
            response_params=response_params_template,
            rng=rng_template,
            current_stats=_empty_lit_stats(),
            history_json="[]",
            warm_start_selected_iteration=0,
        )
        loaded = self._load_lit_continuation_checkpoint(
            template,
            axis=axis,
            state_fingerprint=state_fingerprint,
            full_config_digest=full_config_digest,
        )
        if loaded is None:
            return None
        checkpoint_path, restored = loaded
        if int(restored.axis) != int(axis):
            msg = (
                f"Continuation checkpoint {checkpoint_path} is for axis="
                f"{int(restored.axis)}, expected {axis}."
            )
            raise RuntimeError(msg)

        current_omega = float(restored.current_omega)
        start_omega = self.lit_config.solver.warm_start_omega
        if (
            start_omega is None
            or current_omega < float(start_omega)
            or current_omega >= float(target_omega)
        ):
            msg = (
                f"Continuation checkpoint {checkpoint_path} has current omega "
                f"{current_omega:.8g} outside the resumable interval "
                f"[{start_omega}, {target_omega})."
            )
            raise RuntimeError(msg)

        records = _continuation_records_from_json(
            str(restored.history_json),
            stats_template=restored.current_stats,
        )
        optimized_count = sum(record.optimized for record in records)
        if optimized_count != int(restored.accepted_points):
            msg = (
                f"Continuation checkpoint {checkpoint_path} history contains "
                f"{optimized_count} optimized points, expected "
                f"{int(restored.accepted_points)}."
            )
            raise RuntimeError(msg)
        if not records or not records[-1].optimized:
            msg = (
                f"Continuation checkpoint {checkpoint_path} has no latest-good record."
            )
            raise RuntimeError(msg)
        if not np.isclose(records[-1].omega, current_omega, rtol=0.0, atol=1e-12):
            msg = (
                f"Continuation checkpoint {checkpoint_path} history ends at "
                f"{records[-1].omega:.8g}, not current omega {current_omega:.8g}."
            )
            raise RuntimeError(msg)

        revalidated_stats = jax.device_get(
            self._evaluate_lit_checkpoint(
                response_apply=response_apply,
                response_params=restored.response_params,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                eval_pool=eval_pool,
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=current_omega,
            )
        )
        _require_eligible_lit_checkpoint(
            revalidated_stats,
            context=(
                f"Restored continuation checkpoint at omega={current_omega:.8g} "
                "failed current numerical validation"
            ),
        )
        _require_lit_stage_health(
            revalidated_stats,
            min_reweight_ess_fraction=(
                self.lit_config.continuation.ess_fraction_minimum
            ),
            context=(
                f"Restored continuation checkpoint at omega={current_omega:.8g} "
                "failed current estimator-health validation"
            ),
        )
        records[-1] = records[-1]._replace(stats=revalidated_stats)
        logger.info(
            "Restored axis=%s continuation checkpoint %s omega=%.6f "
            "bridge_points=%d stored_fidelity=%.6f revalidated_fidelity=%.6f "
            "revalidated_ess=%.3f",
            _AXIS_NAMES[axis],
            checkpoint_path,
            current_omega,
            optimized_count,
            float(restored.current_stats.fidelity),
            float(revalidated_stats.fidelity),
            float(revalidated_stats.reweight_ess_fraction),
        )
        return _ContinuationResumeState(
            response_params=restored.response_params,
            rng=restored.rng,
            current_stats=revalidated_stats,
            current_omega=current_omega,
            records=tuple(records),
            warm_start_selected_iteration=int(restored.warm_start_selected_iteration),
        )

    def _save_lit_continuation_checkpoint(
        self,
        response_params,
        current_stats,
        rng,
        current_omega: float,
        records: list[_ContinuationRecord],
        *,
        axis: int,
        target_omega: float,
        ground_checkpoint_step: int,
        ground_energy: float,
        source_center: float,
        source_norm: float,
        response_parity: int,
        state_fingerprint: str,
        full_config_digest: str,
        warm_start_selected_iteration: int,
    ) -> None:
        """Atomically persist one optimized, numerically healthy bridge."""
        if jax.process_index() != 0:
            return
        save_dir = self._continuation_checkpoint_save_dir(axis)
        if save_dir is None:
            return
        accepted_points = sum(record.optimized for record in records)
        if accepted_points <= 0 or not records or not records[-1].optimized:
            return
        checkpoint = _ContinuationCheckpoint(
            schema_version=_CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
            state_fingerprint=state_fingerprint,
            full_config_digest=full_config_digest,
            axis=int(axis),
            target_omega=float(target_omega),
            current_omega=float(current_omega),
            accepted_points=int(accepted_points),
            ground_checkpoint_step=int(ground_checkpoint_step),
            ground_energy=float(ground_energy),
            source_center=float(source_center),
            source_norm=float(source_norm),
            response_parity=int(response_parity),
            response_params=response_params,
            rng=rng,
            current_stats=current_stats,
            history_json=_continuation_records_to_json(records),
            warm_start_selected_iteration=int(warm_start_selected_iteration),
        )
        manager = LITCheckpointManager(
            save_dir,
            prefix=_CONTINUATION_CHECKPOINT_PREFIX,
        )
        checkpoint_path = manager.save(accepted_points - 1, jax.device_get(checkpoint))
        logger.info(
            "Saved axis=%s continuation checkpoint %s omega=%.6f "
            "bridge_points=%d fidelity=%.6f ess=%.3f",
            _AXIS_NAMES[axis],
            checkpoint_path,
            current_omega,
            accepted_points,
            float(current_stats.fidelity),
            float(current_stats.reweight_ess_fraction),
        )

    def _require_continuation_probe_recovery_allowed(
        self,
        current_stats,
        probe_stats,
        *,
        candidate_omega: float,
        min_step_override: bool,
    ) -> None:
        """Reject an unsafe or explicitly disabled minimum-step recovery.

        Raises:
            RuntimeError: If held-out ESS is too low or minimum-step recovery
                was explicitly disabled.
        """
        if not min_step_override:
            return
        probe_ess_failure = _lit_stage_ess_failure(
            probe_stats,
            min_reweight_ess_fraction=(
                self.lit_config.continuation.ess_fraction_minimum
            ),
        )
        if probe_ess_failure is not None:
            msg = (
                "Frequency continuation reached its minimum step with "
                "insufficient held-out importance-sampling ESS; refusing to "
                "optimize from an unreliable estimator at "
                f"omega={candidate_omega:.8g}: "
                f"{probe_ess_failure}."
            )
            raise RuntimeError(msg)
        if self.lit_config.continuation.allow_minimum_step_recovery:
            return
        current_fidelity = float(jax.device_get(current_stats.fidelity))
        candidate_fidelity = float(jax.device_get(probe_stats.fidelity))
        candidate_ess = float(jax.device_get(probe_stats.reweight_ess_fraction))
        required_probe_fidelity = max(
            self.lit_config.continuation.fidelity_retention * current_fidelity,
            0.0,
        )
        msg = (
            "Frequency continuation reached its minimum step without an "
            "acceptable inherited checkpoint and recovery is disabled at "
            f"omega={candidate_omega:.8g}: fidelity={candidate_fidelity:.6f}, "
            f"required relative fidelity={required_probe_fidelity:.6f}, ESS "
            f"fraction={candidate_ess:.6f}, required ESS="
            f"{self.lit_config.continuation.ess_fraction_minimum:.6f}."
        )
        raise RuntimeError(msg)

    def _continue_lit_to_spectrum(
        self,
        update_step,
        response_params,
        current_stats,
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
        target_omega: float,
        spectrum_omega: np.ndarray,
        resume_omega: float | None = None,
        existing_records: tuple[_ContinuationRecord, ...] = (),
        checkpoint_callback: Callable[
            [Any, Any, Any, float, list[_ContinuationRecord]],
            None,
        ]
        | None = None,
    ):
        """Adaptively bridge the warm start to the first reported frequency.

        If ``psi_omega`` solves ``A(omega) psi = Phi``, reusing it at
        ``omega + delta`` gives a relative residual of order
        ``|delta| sqrt(L(omega) / ||Phi||^2)``.  We bound that quantity by the
        configured step fraction and verify every proposed step on the fixed
        held-out pool, bisecting when fidelity degrades too much.

        Returns:
            Parameters prepared for the target, the latest optimized held-out
            statistics, continuation records (including the target probe), and
            the next random key.

        Raises:
            RuntimeError: If a probe is invalid, an optimized bridge checkpoint
                is invalid, or the configured bridge-point cap is exhausted.
        """
        start_omega = (
            float(resume_omega)
            if resume_omega is not None
            else self.lit_config.solver.warm_start_omega
        )
        if start_omega is None or float(start_omega) >= float(target_omega):
            return response_params, current_stats, list(existing_records), rng

        common = dict(
            response_apply=response_apply,
            ground_logpsi=ground_logpsi,
            ground_params=ground_params,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
        )
        current_omega = float(start_omega)
        if current_stats is None:
            current_stats = jax.device_get(
                self._evaluate_lit_checkpoint(
                    response_params=response_params,
                    eval_pool=eval_pool,
                    omega=current_omega,
                    **common,
                )
            )
        _require_eligible_lit_checkpoint(
            current_stats,
            context=(
                "Frequency continuation received an ineligible starting "
                f"checkpoint at omega={current_omega:.8g}"
            ),
        )
        _require_lit_stage_health(
            current_stats,
            min_reweight_ess_fraction=(
                self.lit_config.continuation.ess_fraction_minimum
            ),
            context=(
                "Frequency continuation received an estimator-unhealthy "
                f"starting checkpoint at omega={current_omega:.8g}"
            ),
        )
        min_step = _continuation_min_step(self.lit_config, spectrum_omega)
        records = list(existing_records)
        existing_optimized_count = sum(record.optimized for record in records)
        if existing_optimized_count > self.lit_config.continuation.maximum_points:
            msg = (
                "Restored frequency continuation already contains "
                f"{existing_optimized_count} bridge points, exceeding the current "
                f"limit {self.lit_config.continuation.maximum_points}."
            )
            raise RuntimeError(msg)
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(target_omega)) * 8.0

        def evaluate_probe(omega: float):
            return jax.device_get(
                self._evaluate_lit_checkpoint(
                    response_params=response_params,
                    eval_pool=eval_pool,
                    omega=omega,
                    **common,
                )
            )

        while target_omega - current_omega > tolerance:
            gap = float(target_omega - current_omega)
            physics_step = _physics_continuation_step(
                current_stats,
                gap=gap,
                fraction=self.lit_config.continuation.step_fraction,
                min_step=min_step,
            )
            history_cap = _continuation_history_step_cap(
                records,
                growth_factor=(self.lit_config.continuation.step_growth_factor),
                min_step=min_step,
            )
            step = min(
                physics_step,
                gap if history_cap is None else history_cap,
            )
            optimized_count = sum(record.optimized for record in records)
            capacity = _continuation_capacity_diagnostics(
                remaining_gap=gap,
                optimized_count=optimized_count,
                maximum=self.lit_config.continuation.maximum_points,
                chosen_step=step,
            )
            logger.info(
                "axis=%s continuation_proposal current_omega=%.6f "
                "physics_step=%.6e history_cap=%.6e chosen_step=%.6e "
                "remaining_gap=%.6e optimized_points=%d "
                "remaining_bridge_slots=%d required_mean_step=%.6e "
                "capacity_ratio=%.3f",
                _AXIS_NAMES[axis],
                current_omega,
                physics_step,
                float("nan") if history_cap is None else history_cap,
                step,
                capacity.remaining_gap,
                optimized_count,
                capacity.remaining_bridge_slots,
                capacity.required_mean_step,
                capacity.capacity_ratio,
            )
            candidate_omega = current_omega + step
            probe_stats, candidate_omega, probe_ok, bisections = (
                _bisect_continuation_probe(
                    evaluate_probe,
                    current_stats,
                    current_omega=current_omega,
                    candidate_omega=candidate_omega,
                    target_omega=target_omega,
                    min_step=min_step,
                    retention=(self.lit_config.continuation.fidelity_retention),
                    min_reweight_ess_fraction=(
                        self.lit_config.continuation.ess_fraction_minimum
                    ),
                )
            )

            _require_eligible_lit_checkpoint(
                probe_stats,
                context=(
                    "Frequency continuation produced non-finite/invalid "
                    "held-out statistics at "
                    f"omega={candidate_omega:.8g}; refusing to propagate "
                    "a corrupted checkpoint"
                ),
            )
            actual_step = float(candidate_omega - current_omega)
            min_step_override = not probe_ok and actual_step <= min_step * (1.0 + 1e-12)
            self._require_continuation_probe_recovery_allowed(
                current_stats,
                probe_stats,
                candidate_omega=candidate_omega,
                min_step_override=min_step_override,
            )
            if target_omega - candidate_omega <= tolerance:
                records.append(
                    _ContinuationRecord(
                        omega=float(candidate_omega),
                        optimized=False,
                        selected_iteration=-1,
                        stats=probe_stats,
                        inherited_fidelity=float(probe_stats.fidelity),
                        step=actual_step,
                        bisections=bisections,
                        probe_accepted=probe_ok,
                        min_step_override=min_step_override,
                    )
                )
                logger.info(
                    "axis=%s continuation_probe target=%.6f "
                    "inherited_fidelity=%.6f step=%.6e bisections=%d "
                    "accepted=%s min_step_override=%s",
                    _AXIS_NAMES[axis],
                    target_omega,
                    float(probe_stats.fidelity),
                    actual_step,
                    bisections,
                    probe_ok,
                    min_step_override,
                )
                return response_params, current_stats, records, rng

            _require_continuation_point_capacity(
                optimized_count,
                maximum=self.lit_config.continuation.maximum_points,
                target_omega=target_omega,
            )
            inherited_fidelity = float(probe_stats.fidelity)
            (
                response_params,
                current_stats,
                selected_iteration,
                rng,
            ) = self._optimize_lit_frequency(
                update_step,
                response_params,
                train_pool,
                eval_pool,
                rng,
                omega=candidate_omega,
                iterations=self.lit_config.continuation.iterations,
                stage="continuation",
                **common,
            )

            records.append(
                _ContinuationRecord(
                    omega=float(candidate_omega),
                    optimized=True,
                    selected_iteration=selected_iteration,
                    stats=current_stats,
                    inherited_fidelity=inherited_fidelity,
                    step=actual_step,
                    bisections=bisections,
                    probe_accepted=probe_ok,
                    min_step_override=min_step_override,
                )
            )
            logger.info(
                "axis=%s continuation_step omega=%.6f inherited_fidelity=%.6f "
                "selected_fidelity=%.6f step=%.6e bisections=%d accepted=%s "
                "min_step_override=%s",
                _AXIS_NAMES[axis],
                candidate_omega,
                inherited_fidelity,
                float(current_stats.fidelity),
                actual_step,
                bisections,
                probe_ok,
                min_step_override,
            )
            current_omega = candidate_omega
            if checkpoint_callback is not None:
                checkpoint_callback(
                    response_params,
                    current_stats,
                    rng,
                    current_omega,
                    records,
                )

        return response_params, current_stats, records, rng
