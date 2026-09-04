# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Adaptive continuation and durable-state helpers for LIT."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict
from enum import Enum
from typing import Any, cast
from zipfile import BadZipFile

import jax
import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc_contrib_lit.common import (
    _CONTINUATION_CHECKPOINT_PREFIX,
    _CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
)
from jaqmc_contrib_lit.config import LITConfig
from jaqmc_contrib_lit.optimization import (
    _is_eligible_lit_checkpoint,
    _lit_stage_ess_failure,
)
from jaqmc_contrib_lit.response import (
    LITStats,
)
from jaqmc_contrib_lit.state import (
    _ContinuationCapacityDiagnostics,
    _ContinuationRecord,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


def _continuation_min_step(
    config: LITConfig,
    spectrum_omega: np.ndarray,
) -> float:
    configured = config.continuation.minimum_step
    if configured is not None:
        return float(configured)
    candidates = [float(config.continuation.step_fraction) * float(config.eta)]
    spacings = np.diff(np.asarray(spectrum_omega, dtype=np.float64))
    if spacings.size:
        candidates.append(float(np.min(spacings)))
    return max(np.finfo(np.float64).eps, min(candidates))


def _bisect_continuation_probe(
    evaluate: Callable[[float], Any],
    current_stats,
    *,
    current_omega: float,
    candidate_omega: float,
    target_omega: float,
    min_step: float,
    retention: float,
    min_reweight_ess_fraction: float,
):
    """Bisect an inherited-parameter probe until it is safe or minimal.

    Returns:
        Probe statistics, accepted frequency, acceptance flag, and bisections.
    """
    bisections = 0
    while True:
        probe_stats = evaluate(candidate_omega)
        probe_ok = _continuation_probe_is_acceptable(
            current_stats,
            probe_stats,
            retention=retention,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        candidate_gap = candidate_omega - current_omega
        if probe_ok or candidate_gap <= min_step * (1.0 + 1e-12):
            return probe_stats, candidate_omega, probe_ok, bisections
        candidate_gap = max(min_step, 0.5 * candidate_gap)
        candidate_omega = min(target_omega, current_omega + candidate_gap)
        bisections += 1


def _require_continuation_point_capacity(
    optimized_count: int,
    *,
    maximum: int,
    target_omega: float,
) -> None:
    """Reject an additional optimized bridge after the configured cap.

    Raises:
        RuntimeError: If no additional bridge point may be optimized.
    """
    if optimized_count >= maximum:
        msg = (
            "Adaptive frequency continuation exceeded "
            f"{maximum} bridge points before omega={target_omega:.8g}."
        )
        raise RuntimeError(msg)


def _continuation_capacity_diagnostics(
    *,
    remaining_gap: float,
    optimized_count: int,
    maximum: int,
    chosen_step: float,
) -> _ContinuationCapacityDiagnostics:
    """Forecast whether one proposal pace fits the remaining point budget.

    The forecast is diagnostic only.  It assumes ``chosen_step`` is held for
    every remaining optimized bridge plus the final unoptimized target probe;
    later growth or bisection will change the realized trajectory.

    Returns:
        The remaining budget and proposal pace relative to its required mean.

    Raises:
        ValueError: If the gap, step, or point counts are invalid.
    """
    remaining_gap = float(remaining_gap)
    chosen_step = float(chosen_step)
    optimized_count = int(optimized_count)
    maximum = int(maximum)
    if not np.isfinite(remaining_gap) or remaining_gap <= 0.0:
        msg = f"remaining_gap must be finite and positive, got {remaining_gap!r}."
        raise ValueError(msg)
    if not np.isfinite(chosen_step) or chosen_step <= 0.0:
        msg = f"chosen_step must be finite and positive, got {chosen_step!r}."
        raise ValueError(msg)
    if optimized_count < 0 or maximum < 0 or optimized_count > maximum:
        msg = (
            "Continuation point counts must satisfy "
            f"0 <= optimized_count <= maximum, got {optimized_count} and {maximum}."
        )
        raise ValueError(msg)
    remaining_bridge_slots = maximum - optimized_count
    required_mean_step = remaining_gap / (remaining_bridge_slots + 1)
    return _ContinuationCapacityDiagnostics(
        remaining_gap=remaining_gap,
        remaining_bridge_slots=remaining_bridge_slots,
        required_mean_step=required_mean_step,
        capacity_ratio=chosen_step / required_mean_step,
    )


def _continuation_history_step_cap(
    records: list[_ContinuationRecord],
    *,
    growth_factor: float,
    min_step: float,
) -> float | None:
    """Recover a conservative next-step cap from accepted bridge history.

    A bridge that needed any bisection, failed its inherited probe, or used a
    minimum-step override holds its actual accepted step.  Only a completely
    clean bridge may grow the cap.  Because the accepted record is already in
    the durable checkpoint history, interrupted and uninterrupted runs make
    the same next proposal without adding controller state to the schema.

    Returns:
        The next history-derived step cap, or ``None`` without an accepted
        bridge.

    Raises:
        RuntimeError: If the latest accepted record contains an invalid step.
    """
    latest = next((record for record in reversed(records) if record.optimized), None)
    if latest is None:
        return None
    accepted_step = float(latest.step)
    if not np.isfinite(accepted_step) or accepted_step <= 0.0:
        msg = (
            "Latest optimized continuation record has an invalid accepted "
            f"step {accepted_step!r} at omega={float(latest.omega):.8g}."
        )
        raise RuntimeError(msg)
    clean_success = (
        int(latest.bisections) == 0
        and bool(latest.probe_accepted)
        and not bool(latest.min_step_override)
    )
    multiplier = float(growth_factor) if clean_success else 1.0
    return max(float(min_step), multiplier * accepted_step)


def _physics_continuation_step(stats, *, gap: float, fraction: float, min_step: float):
    """Choose a homotopy step from the inherited LIT residual estimate.

    Returns:
        A positive step no larger than the remaining target gap.
    """
    signed_lit = float(jax.device_get(stats.signed_lit))
    source_norm = float(jax.device_get(stats.source_norm))
    if (
        np.isfinite(signed_lit)
        and np.isfinite(source_norm)
        and signed_lit > 0.0
        and source_norm > 0.0
    ):
        proposed = float(fraction) * np.sqrt(source_norm / signed_lit)
    else:
        proposed = float(min_step)
    return min(float(gap), max(float(min_step), proposed))


def _continuation_probe_is_acceptable(
    current,
    candidate,
    *,
    retention: float,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    """Return whether an inherited checkpoint is safe enough to optimize.

    This relative check protects initialization quality without imposing an
    arbitrary absolute fidelity threshold on the optimized result.
    """
    if not _is_eligible_lit_checkpoint(candidate):
        return False
    if (
        _lit_stage_ess_failure(
            candidate,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        is not None
    ):
        return False
    current_fidelity = float(jax.device_get(current.fidelity))
    candidate_fidelity = float(jax.device_get(candidate.fidelity))
    if not np.isfinite(current_fidelity):
        return False
    required = max(0.0, float(retention) * current_fidelity)
    return candidate_fidelity >= required


_CONTINUATION_HISTORY_STAT_FIELDS = (
    "fidelity",
    "reverse_kl",
    "invalid_sample_fraction",
)


def _empty_lit_stats() -> LITStats:
    return LITStats(
        **{
            name: jnp.asarray(False if name == "error_d_valid" else 0.0)
            for name in LITStats._fields
        }
    )


def _checkpoint_json_value(value):
    if isinstance(value, Enum):
        return _checkpoint_json_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _checkpoint_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_checkpoint_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _checkpoint_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _checkpoint_json_value(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _tree_shape_dtype_signature(tree) -> dict[str, object]:
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(tree)
    leaves = []
    for key_path, leaf in leaves_with_path:
        shape = tuple(int(size) for size in getattr(leaf, "shape", ()))
        dtype = getattr(leaf, "dtype", type(leaf).__name__)
        leaves.append(
            {
                "path": str(key_path),
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )
    return {"treedef": str(treedef), "leaves": leaves}


def _tree_content_digest(tree) -> str:
    """Return a deterministic SHA-256 digest of one concrete PyTree.

    Checkpoint compatibility must distinguish equal-shaped ground states and
    source pools belonging to different systems or runs.  Fresh response
    parameters intentionally use only their shape/dtype signature because
    they are a restore template and are replaced by the saved parameters.

    Raises:
        TypeError: If a leaf uses an object dtype without stable byte content.
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(tree)
    digest = hashlib.sha256(str(treedef).encode("utf-8"))
    for key_path, leaf in leaves_with_path:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            msg = (
                "Continuation checkpoint fingerprints cannot hash object-dtype "
                f"leaf {key_path}."
            )
            raise TypeError(msg)
        metadata = json.dumps(
            {
                "path": str(key_path),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _source_pool_target_digest(ground_params, molecule_data) -> str:
    """Bind a source pool to the density and Hamiltonian that define pi_Phi.

    Returns:
        A stable hexadecimal digest of the ground parameters and static
        molecular geometry.
    """
    return _canonical_sha256(
        {
            "schema_version": 1,
            "ground_params_sha256": _tree_content_digest(ground_params),
            "atoms_sha256": _tree_content_digest(molecule_data.atoms),
            "charges_sha256": _tree_content_digest(molecule_data.charges),
        }
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        _checkpoint_json_value(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _continuation_checkpoint_digests(
    config: LITConfig,
    *,
    response_params,
    ground_params,
    train_pool,
    eval_pool,
    axis: int,
    source_center: float,
    source_norm: float,
    ground_energy: float,
    ground_checkpoint_step: int,
    response_parity: int,
    target_omega: float,
    spectrum_omega: np.ndarray,
) -> tuple[str, str]:
    """Return compatibility and full-audit digests for continuation state."""
    # configurable_dataclass creates real dataclasses at runtime, but the
    # external decorator does not currently expose a dataclass transform.
    config_payload = asdict(cast(Any, config))
    config_payload["continuation"].pop("restore_path")
    state_config = {
        "eta": config.eta,
        "axes": config.axes,
        "omega": asdict(cast(Any, config.omega)),
        "ground": {
            "allow_untrained": config.ground.allow_untrained,
            "energy": config.ground.energy,
        },
        "source": {
            "center_override": config.source.center_override,
            "norm_override": config.source.norm_override,
            "floor": config.source.floor,
        },
        "ansatz": {
            "determinants": config.ansatz.determinants,
            "hidden_dims_single": config.ansatz.hidden_dims_single,
            "hidden_dims_double": config.ansatz.hidden_dims_double,
            "use_last_layer": config.ansatz.use_last_layer,
            "envelope": config.ansatz.envelope,
            "orbitals_spin_split": config.ansatz.orbitals_spin_split,
            "sector_tolerance": config.ansatz.sector_tolerance,
        },
    }
    dynamic_payload = {
        "schema_version": _CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
        "axis": int(axis),
        "source_center": float(source_center),
        "source_norm": float(source_norm),
        "ground_energy": float(ground_energy),
        "ground_checkpoint_step": int(ground_checkpoint_step),
        "response_parity": int(response_parity),
        "target_omega": float(target_omega),
        "spectrum_omega": np.asarray(spectrum_omega, dtype=np.float64),
        "response_params": _tree_shape_dtype_signature(response_params),
        "ground_params": {
            "signature": _tree_shape_dtype_signature(ground_params),
            "content_sha256": _tree_content_digest(ground_params),
        },
        # Pool contents include the sampled electron configurations and the
        # static molecular data, so they bind a resume to both its held-out
        # population and its Hamiltonian/system identity.
        "train_pool_sha256": _tree_content_digest(train_pool),
        "eval_pool_sha256": _tree_content_digest(eval_pool),
    }
    state_fingerprint = _canonical_sha256(
        {"config": state_config, "dynamic": dynamic_payload}
    )
    full_config_digest = _canonical_sha256(
        {"config": config_payload, "dynamic": dynamic_payload}
    )
    return state_fingerprint, full_config_digest


def _continuation_records_to_json(records: list[_ContinuationRecord]) -> str:
    payload = []
    for record in records:
        stats_payload = {
            name: float(jax.device_get(getattr(record.stats, name, 0.0)))
            for name in _CONTINUATION_HISTORY_STAT_FIELDS
        }
        payload.append(
            {
                "omega": float(record.omega),
                "optimized": bool(record.optimized),
                "selected_iteration": int(record.selected_iteration),
                "stats": stats_payload,
                "inherited_fidelity": float(record.inherited_fidelity),
                "step": float(record.step),
                "bisections": int(record.bisections),
                "probe_accepted": bool(record.probe_accepted),
                "min_step_override": bool(record.min_step_override),
            }
        )
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _continuation_records_from_json(
    encoded: str,
    *,
    stats_template: LITStats,
) -> list[_ContinuationRecord]:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Continuation checkpoint history is not valid JSON."
        ) from exc
    if not isinstance(payload, list):
        raise RuntimeError("Continuation checkpoint history must be a JSON list.")
    records = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("stats"), dict):
            raise RuntimeError("Continuation checkpoint history entry is malformed.")
        replacements = {
            name: jnp.asarray(float(item["stats"][name]))
            for name in _CONTINUATION_HISTORY_STAT_FIELDS
        }
        records.append(
            _ContinuationRecord(
                omega=float(item["omega"]),
                optimized=bool(item["optimized"]),
                selected_iteration=int(item["selected_iteration"]),
                stats=stats_template._replace(**replacements),
                inherited_fidelity=float(item["inherited_fidelity"]),
                step=float(item["step"]),
                bisections=int(item["bisections"]),
                probe_accepted=bool(item["probe_accepted"]),
                min_step_override=bool(item["min_step_override"]),
            )
        )
    return records


def _read_continuation_checkpoint_metadata(path: UPath) -> dict[str, object]:
    with path.open("rb") as f_in, np.load(f_in, allow_pickle=False) as npf:
        return {
            "schema_version": int(npf["schema_version"].item()),
            "state_fingerprint": str(npf["state_fingerprint"].item()),
            "full_config_digest": str(npf["full_config_digest"].item()),
        }


def _latest_continuation_checkpoint_path(path: UPath) -> UPath | None:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            path.glob(f"{_CONTINUATION_CHECKPOINT_PREFIX}_ckpt_*.npz"),
            reverse=True,
        )
    else:
        return None
    for candidate in candidates:
        try:
            _read_continuation_checkpoint_metadata(candidate)
        except (OSError, EOFError, BadZipFile, KeyError, ValueError):
            logger.warning("Ignoring unreadable continuation checkpoint %s", candidate)
            continue
        return candidate
    return None
