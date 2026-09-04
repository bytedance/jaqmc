# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from jaqmc_contrib_lit.inversion import forward_lit
from jaqmc_contrib_lit.postprocess import LITInversionPostprocessor

from jaqmc.utils.config import ConfigManager


def _write_single_pole_lit(
    path: Path,
    *,
    axes: str = "x",
    strengths: tuple[float, ...] = (1.25,),
) -> None:
    omega = np.linspace(0.75, 0.90, 81)
    eta = 0.003
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=np.asarray([0.78]),
        pole_strengths=np.asarray(strengths)[:, np.newaxis],
        background_coefficients=np.asarray(
            [[0.01 + 0.01 * index] for index in range(len(strengths))]
        ),
    )
    phase = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    profile = np.linspace(-1.0, 1.0, omega.size)
    blocks = (
        signed_lit[..., np.newaxis]
        + 1e-5
        * profile[np.newaxis, :, np.newaxis]
        * np.sin(phase)[np.newaxis, np.newaxis, :]
    )
    np.savez(
        path,
        omega=omega,
        eta=eta,
        axes=axes,
        axis_indices=np.asarray([{"x": 0, "y": 1, "z": 2}[axis] for axis in axes]),
        signed_lit=signed_lit,
        signed_lit_jackknife_blocks=blocks,
        signed_lit_jackknife_block_count=blocks.shape[-1],
        eval_pool_sha256=np.asarray([f"pool-{axis}" for axis in axes]),
        error_bound_monitor=np.full_like(signed_lit, 1e-6),
        error_d_valid=np.ones_like(signed_lit, dtype=np.bool_),
    )


def test_manual_postprocessor_runs_article_fit_and_writes_separate_output(
    tmp_path: Path,
):
    input_path = tmp_path / "lit_spectrum.npz"
    output_path = tmp_path / "inversion_k1.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "output_path": str(output_path),
                "pole_count": 1,
                "background_order": 0,
                "pole_fit_tolerance": 1e-9,
            }
        }
    )

    LITInversionPostprocessor(cfg)()

    assert input_path.exists()
    assert output_path.exists()
    with np.load(output_path, allow_pickle=False) as result:
        assert result["manual_postprocess"].item()
        assert result["requested_pole_count"].item() == 1
        assert result["pole_initialization_method"].item() == "ordinary_ls_greedy"
        assert result["method"].item() == "article_single_width_discrete_line_fit"
        assert result["background_order"].item() == 0
        assert not result["statistically_weighted"].item()
        assert not result["oscillator_strengths_available"].item()
        np.testing.assert_allclose(result["pole_energies"], [0.78], atol=2e-7)
        np.testing.assert_allclose(
            result["transition_strengths_I_an"],
            [[1.25]],
            atol=1e-8,
        )


def test_manual_postprocessor_outputs_xyz_oscillator_strength(tmp_path: Path):
    input_path = tmp_path / "lit_spectrum_xyz.npz"
    output_path = tmp_path / "line_fit_xyz.npz"
    strengths = (0.1, 0.2, 0.3)
    _write_single_pole_lit(input_path, axes="xyz", strengths=strengths)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "output_path": str(output_path),
                "pole_energies": [0.779],
                "pole_energy_bounds": [[0.77, 0.79]],
                "background_order": 0,
                "pole_fit_tolerance": 1e-9,
            }
        }
    )

    LITInversionPostprocessor(cfg)()

    with np.load(output_path, allow_pickle=False) as result:
        assert result["oscillator_strengths_available"].item()
        expected = (2.0 / 3.0) * 0.78 * sum(strengths)
        np.testing.assert_allclose(
            result["oscillator_strengths"],
            [expected],
            atol=1e-8,
        )


def test_manual_postprocessor_requires_an_explicit_model_hypothesis(tmp_path: Path):
    input_path = tmp_path / "lit_spectrum.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
            }
        }
    )

    with pytest.raises(ValueError, match="article line model is empty"):
        LITInversionPostprocessor(cfg)


def test_manual_postprocessor_never_overwrites_raw_input(tmp_path: Path):
    input_path = tmp_path / "lit_spectrum.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "output_path": str(input_path),
                "pole_energies": [0.78],
            }
        }
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        LITInversionPostprocessor(cfg)
