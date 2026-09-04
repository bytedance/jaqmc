# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""Lorentz integral transform helpers for direct NQS response calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lit_from_poles(
    omega: ArrayLike,
    excitation_energies: ArrayLike,
    strengths: ArrayLike,
    eta: float,
) -> NDArray[np.float64]:
    """Evaluate ``sum_n strengths_n / ((Omega_n - omega)^2 + eta^2)``."""
    if eta <= 0:
        msg = f"eta must be positive, got {eta}"
        raise ValueError(msg)
    omega_arr = np.asarray(omega, dtype=np.float64)
    poles = np.asarray(excitation_energies, dtype=np.float64)
    strengths_arr = np.asarray(strengths, dtype=np.float64)
    if poles.ndim != 1:
        msg = f"excitation_energies must be one-dimensional, got {poles.shape}"
        raise ValueError(msg)
    if strengths_arr.shape[0] != poles.shape[0]:
        msg = (
            "strengths first dimension must match excitation energies, got "
            f"{strengths_arr.shape} and {poles.shape}"
        )
        raise ValueError(msg)
    kernel = 1 / ((poles - omega_arr[..., np.newaxis]) ** 2 + float(eta) ** 2)
    return np.einsum("...n,n...->...", kernel, strengths_arr)


def broadened_from_lit(lit: ArrayLike, eta: float) -> NDArray[np.float64]:
    """Convert a LIT value to the corresponding Lorentzian-convolved spectrum."""
    if eta <= 0:
        msg = f"eta must be positive, got {eta}"
        raise ValueError(msg)
    return float(eta) * np.asarray(lit, dtype=np.float64) / np.pi


def lit_error_bound(
    fidelity: float,
    phi_norm: float,
    normalization_abs: float,
    eta: float,
    d_factor: float = 1.0,
) -> float:
    """Evaluate the LIT fidelity-based error monitor."""
    if not 0 < fidelity <= 1:
        msg = f"fidelity must be in (0, 1], got {fidelity}"
        raise ValueError(msg)
    if phi_norm < 0:
        msg = f"phi_norm must be nonnegative, got {phi_norm}"
        raise ValueError(msg)
    if normalization_abs <= 0:
        msg = f"normalization_abs must be positive, got {normalization_abs}"
        raise ValueError(msg)
    if eta <= 0:
        msg = f"eta must be positive, got {eta}"
        raise ValueError(msg)
    return float(
        d_factor
        * phi_norm
        / (float(eta) * normalization_abs)
        * np.sqrt((1.0 - fidelity) / fidelity)
    )
