# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit CPU postprocessing for the article's saved-LIT line fit."""

from __future__ import annotations

import logging
from dataclasses import field
from pathlib import Path
from typing import SupportsInt, cast

import numpy as np
from numpy.typing import ArrayLike
from upath import UPath

from jaqmc.utils.config import ConfigManager, configurable_dataclass
from jaqmc_contrib_lit.inversion import initialize_lit_poles
from jaqmc_contrib_lit.inversion_io import (
    LITInversionSettings,
    aggregate_lit_npz,
    invert_lit_npz,
    lit_inversion_npz_payload,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


@configurable_dataclass
class LITInversionPostprocessConfig:
    """Configuration for the revision-2 article's single-width line fit.

    This postprocessor remains separate from ``lit run``. The GPU run
    writes only the raw transform, which can be refitted with different local
    windows, pole counts, and polynomial background orders.
    """

    input_paths: tuple[str, ...] = field(default_factory=tuple)
    output_path: str = "lit_inversion.npz"
    assume_independent: bool = False
    require_determined: bool = True

    # A pole count is a model-order hypothesis.  Initial centers are selected
    # from the saved transform with the same ordinary least-squares objective.
    pole_count: int = 0
    pole_search_energy_min: float | None = None
    pole_search_energy_max: float | None = None
    pole_search_grid_points: int = 0
    pole_minimum_separation: float | None = None

    # Alternatively, provide ordered initial centers and optional fit bounds.
    pole_energies: tuple[float, ...] = field(default_factory=tuple)
    pole_energy_bounds: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    # The article uses the lowest polynomial order that gives stable results.
    background_order: int = 0
    fit_omega_min: float | None = None
    fit_omega_max: float | None = None

    pole_fit_tolerance: float = 1e-7
    pole_fit_max_iterations: int = 200
    solver_tolerance: float = 1e-10
    solver_max_iterations: int | None = None


class LITInversionPostprocessor:
    """Load raw LIT NPZs and write one article-aligned line-fit NPZ."""

    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg
        self.config = cfg.get("inversion", LITInversionPostprocessConfig)
        self._validate_config()

    def __call__(self, dry_run: bool = False) -> None:
        self.cfg.finalize()
        if not dry_run:
            self.run()

    def _validate_config(self) -> None:  # noqa: C901
        config = self.config
        paths = config.input_paths
        if not paths or any(not isinstance(path, str) or not path for path in paths):
            raise ValueError(
                "inversion.input_paths must contain at least one nonempty path."
            )
        normalized_inputs = tuple(Path(path).expanduser().resolve() for path in paths)
        if len(set(normalized_inputs)) != len(normalized_inputs):
            raise ValueError("inversion.input_paths must be unique.")
        if not config.output_path or not config.output_path.endswith(".npz"):
            raise ValueError("inversion.output_path must be a nonempty '.npz' path.")
        normalized_output = Path(config.output_path).expanduser().resolve()
        if normalized_output in normalized_inputs:
            raise ValueError(
                "inversion.output_path must not overwrite a raw LIT input."
            )

        pole_count = _nonnegative_integer(config.pole_count, "pole_count")
        grid_points = _nonnegative_integer(
            config.pole_search_grid_points,
            "pole_search_grid_points",
        )
        _nonnegative_integer(config.background_order, "background_order")
        if pole_count and config.pole_energies:
            raise ValueError(
                "inversion.pole_count and inversion.pole_energies are exclusive."
            )
        if pole_count and config.pole_energy_bounds:
            raise ValueError(
                "data-driven pole initialization generates its own bounds."
            )
        if not pole_count and not config.pole_energies:
            raise ValueError(
                "The article line model is empty; set inversion.pole_count or "
                "inversion.pole_energies."
            )
        if pole_count and 0 < grid_points < max(4 * pole_count + 1, 17):
            raise ValueError(
                "inversion.pole_search_grid_points is too small for pole_count."
            )
        for name in (
            "pole_search_energy_min",
            "pole_search_energy_max",
            "pole_minimum_separation",
            "fit_omega_min",
            "fit_omega_max",
        ):
            value = getattr(config, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"inversion.{name} must be finite or null.")
        if (
            config.pole_minimum_separation is not None
            and config.pole_minimum_separation <= 0.0
        ):
            raise ValueError(
                "inversion.pole_minimum_separation must be positive or null."
            )
        for lower_name, upper_name in (
            ("pole_search_energy_min", "pole_search_energy_max"),
            ("fit_omega_min", "fit_omega_max"),
        ):
            lower = getattr(config, lower_name)
            upper = getattr(config, upper_name)
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError(
                    f"inversion.{lower_name} must be below inversion.{upper_name}."
                )

    def run(self) -> None:
        config = self.config
        paths = tuple(config.input_paths)
        data = aggregate_lit_npz(
            paths,
            assume_independent=config.assume_independent,
        )
        fit_mask = np.ones(data.omega.size, dtype=np.bool_)
        if config.fit_omega_min is not None:
            fit_mask &= data.omega >= config.fit_omega_min
        if config.fit_omega_max is not None:
            fit_mask &= data.omega <= config.fit_omega_max
        if not np.any(fit_mask):
            raise ValueError("the configured fit window contains no observations")

        pole_initialization = None
        pole_energies: ArrayLike
        pole_bounds: ArrayLike
        if config.pole_count > 0:
            pole_initialization = initialize_lit_poles(
                data.omega[fit_mask],
                data.eta[fit_mask],
                data.signed_lit[:, fit_mask],
                pole_count=config.pole_count,
                background_order=config.background_order,
                energy_min=config.pole_search_energy_min,
                energy_max=config.pole_search_energy_max,
                candidate_grid_points=(config.pole_search_grid_points or None),
                minimum_separation=config.pole_minimum_separation,
                solver_tolerance=config.solver_tolerance,
                solver_max_iterations=config.solver_max_iterations,
            )
            pole_energies = pole_initialization.pole_energies
            pole_bounds = pole_initialization.pole_energy_bounds
            logger.info(
                "Initialized K=%d article line model: energies=%s Q=%.10e",
                config.pole_count,
                np.array2string(pole_energies, precision=10),
                pole_initialization.objective,
            )
        else:
            pole_energies = config.pole_energies
            pole_bounds = config.pole_energy_bounds

        settings = LITInversionSettings(
            pole_energies=pole_energies,
            pole_energy_bounds=(
                None
                if np.asarray(pole_bounds, dtype=np.float64).size == 0
                else pole_bounds
            ),
            background_order=config.background_order,
            fit_omega_min=config.fit_omega_min,
            fit_omega_max=config.fit_omega_max,
            pole_fit_tolerance=config.pole_fit_tolerance,
            pole_fit_max_iterations=config.pole_fit_max_iterations,
            solver_tolerance=config.solver_tolerance,
            solver_max_iterations=config.solver_max_iterations,
        )
        inversion = invert_lit_npz(
            paths,
            settings,
            assume_independent=config.assume_independent,
        )
        diagnostics = inversion.result.diagnostics
        if config.require_determined and diagnostics.underdetermined:
            reasons = "; ".join(diagnostics.underdetermined_reasons)
            raise RuntimeError(f"Article LIT line fit is underdetermined: {reasons}")

        payload = lit_inversion_npz_payload(inversion)
        payload["manual_postprocess"] = np.asarray(True)
        payload["requested_pole_count"] = np.asarray(
            config.pole_count,
            dtype=np.int64,
        )
        payload["pole_initialization_method"] = np.asarray(
            "ordinary_ls_greedy" if pole_initialization is not None else "configured"
        )
        if pole_initialization is not None:
            payload["pole_initialization_objective"] = np.asarray(
                pole_initialization.objective,
                dtype=np.float64,
            )
            payload["pole_initialization_candidate_grid_points"] = np.asarray(
                pole_initialization.candidate_grid_points,
                dtype=np.int64,
            )
            payload["pole_initialization_minimum_separation"] = np.asarray(
                pole_initialization.minimum_separation,
                dtype=np.float64,
            )

        output_path = UPath(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f_out:
            np.savez_compressed(f_out, **payload)  # type: ignore[arg-type]
        logger.info(
            "Wrote article single-width line fit to %s (poles=%s, "
            "background_order=%d, oscillator_strengths=%s)",
            output_path,
            np.array2string(inversion.result.pole_energies, precision=10),
            config.background_order,
            bool(payload["oscillator_strengths_available"]),
        )


def _nonnegative_integer(value: object, name: str) -> int:
    try:
        converted = int(cast(SupportsInt, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"inversion.{name} must be a nonnegative integer.") from error
    if isinstance(value, (bool, np.bool_)) or converted != value or converted < 0:
        raise ValueError(f"inversion.{name} must be a nonnegative integer.")
    return converted


__all__ = [
    "LITInversionPostprocessConfig",
    "LITInversionPostprocessor",
]
