# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers and constants for LIT workflows."""

from __future__ import annotations

import logging

import numpy as np
from upath import UPath

from jaqmc_contrib_lit.config import LITConfig

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)

_ATOM_PARITY_PENDING_SECTOR_LABEL = "atom_parity_pending"
_ATOM_HARD_ODD_SECTOR_LABEL = "atom_odd_hard"
_ATOM_HARD_EVEN_SECTOR_LABEL = "atom_even_hard"
_CONTINUATION_CHECKPOINT_SCHEMA_VERSION = 3
_CONTINUATION_CHECKPOINT_PREFIX = "continuation"


_AXIS_NAMES = ("x", "y", "z")


def _lit_omega_grid(config: LITConfig) -> np.ndarray:
    if config.omega.values:
        omega = np.asarray(tuple(float(value) for value in config.omega.values))
        if omega.ndim != 1 or omega.size == 0:
            msg = "lit.omega.values must be a non-empty one-dimensional sequence."
            raise ValueError(msg)
        if not np.all(np.isfinite(omega)):
            msg = "lit.omega.values must contain only finite values."
            raise ValueError(msg)
        if np.any(np.diff(omega) <= 0.0):
            msg = "lit.omega.values must be strictly increasing."
            raise ValueError(msg)
        return omega
    if config.omega.points < 1:
        msg = "lit.omega.points must be positive."
        raise ValueError(msg)
    if not np.isfinite(config.omega.minimum) or not np.isfinite(config.omega.maximum):
        msg = "lit.omega.minimum and lit.omega.maximum must be finite."
        raise ValueError(msg)
    if config.omega.points > 1 and config.omega.maximum <= config.omega.minimum:
        msg = "lit.omega.maximum must exceed lit.omega.minimum for a serial scan."
        raise ValueError(msg)
    return np.linspace(
        config.omega.minimum,
        config.omega.maximum,
        config.omega.points,
    )


def _three_component_override(
    value: float | tuple[float, float, float] | None,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray | None:
    """Normalize a scalar or Cartesian override to a length-three host array.

    Returns:
        ``None`` when unset, otherwise a finite float64 array of shape ``(3,)``.

    Raises:
        ValueError: If the override has the wrong shape or invalid values.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(3, float(array), dtype=np.float64)
    elif array.shape == (3,):
        array = np.array(array, dtype=np.float64, copy=True)
    else:
        msg = f"{name} must be a scalar or length-three Cartesian vector."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    if positive and np.any(array <= 0.0):
        msg = f"{name} must contain only positive values."
        raise ValueError(msg)
    return array


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _two_spin_tuple(values) -> tuple[int, int]:
    nspins = tuple(int(value) for value in values)
    if len(nspins) != 2:
        msg = f"Expected two spin populations, got {nspins}."
        raise ValueError(msg)
    return nspins


def _save_npz(path: UPath, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f_out:
        np.savez(f_out, **payload)  # type: ignore[arg-type]


def _axis_indices(axes: str) -> tuple[int, ...]:
    lookup = {name: idx for idx, name in enumerate(_AXIS_NAMES)}
    result = []
    for raw in axes.lower():
        if raw not in lookup:
            msg = f"Unknown dipole axis {raw!r}; expected characters from 'xyz'."
            raise ValueError(msg)
        result.append(lookup[raw])
    if not result:
        msg = "At least one dipole axis is required."
        raise ValueError(msg)
    if len(result) != len(set(result)):
        msg = f"Duplicate dipole axes are not allowed: {axes!r}."
        raise ValueError(msg)
    return tuple(result)
