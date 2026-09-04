# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="arg-type"

from pathlib import Path

import numpy as np
import pytest
from jaqmc_contrib_lit.inversion import lit_block_statistics
from jaqmc_contrib_lit.inversion_io import (
    aggregate_lit_npz,
)


def _write_workflow_npz(
    path: Path,
    *,
    omega: np.ndarray,
    eta: float,
    signed_lit: np.ndarray,
    blocks: np.ndarray | None = None,
    raw_blocks: np.ndarray | None = None,
    digests: tuple[str, ...] | str | None = ("pool-x", "pool-z"),
    covariance: np.ndarray | None = None,
    systematic_error: np.ndarray | None = None,
    error_d_valid: np.ndarray | None = None,
    axes: str = "xz",
    axis_indices: np.ndarray = np.asarray([0, 2]),
) -> None:
    payload: dict[str, object] = {
        "omega": omega,
        "eta": eta,
        "signed_lit": signed_lit,
        "axes": axes,
        "axis_indices": axis_indices,
        "error_bound_monitor": (
            np.zeros_like(signed_lit) if systematic_error is None else systematic_error
        ),
        "error_d_valid": (
            np.ones_like(signed_lit, dtype=np.bool_)
            if error_d_valid is None
            else error_d_valid
        ),
    }
    if blocks is not None:
        payload["signed_lit_jackknife_blocks"] = blocks
        payload["signed_lit_jackknife_block_count"] = blocks.shape[-1]
    if raw_blocks is not None:
        payload["signed_lit_blocks"] = raw_blocks
    if digests is not None:
        payload["eval_pool_sha256"] = np.asarray(digests)
    if covariance is not None:
        payload["signed_lit_covariance"] = covariance
    np.savez(path, **payload)


def test_aggregate_matched_blocks_preserves_cross_eta_covariance(tmp_path: Path):
    omega_a = np.asarray([0.1, 0.2])
    omega_b = np.asarray([0.15, 0.25, 0.35])
    full_a = np.asarray([[10.0, 11.0], [20.0, 21.0]])
    full_b = np.asarray([[12.0, 13.0, 14.0], [22.0, 23.0, 24.0]])
    blocks_a = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5) / 10.0
    blocks_b = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5) / 7.0 + 2.0
    systematic_a = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    systematic_b = np.asarray([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    path_a = tmp_path / "eta-a.npz"
    path_b = tmp_path / "eta-b.npz"
    _write_workflow_npz(
        path_a,
        omega=omega_a,
        eta=0.03,
        signed_lit=full_a,
        blocks=blocks_a,
        systematic_error=systematic_a,
    )
    _write_workflow_npz(
        path_b,
        omega=omega_b,
        eta=0.08,
        signed_lit=full_b,
        blocks=blocks_b,
        systematic_error=systematic_b,
    )

    result = aggregate_lit_npz([path_a, path_b])

    combined_blocks = np.concatenate([blocks_a, blocks_b], axis=1)
    expected_statistical = lit_block_statistics(combined_blocks).covariance
    expected_systematic = np.concatenate([systematic_a, systematic_b], axis=1)
    expected_covariance = np.array(expected_statistical, copy=True)
    diagonal = np.arange(expected_covariance.shape[-1])
    expected_covariance[:, diagonal, diagonal] += expected_systematic**2
    np.testing.assert_array_equal(result.omega, np.concatenate([omega_a, omega_b]))
    np.testing.assert_array_equal(
        result.eta,
        np.asarray([0.03, 0.03, 0.08, 0.08, 0.08]),
    )
    # The formal central value is the full-pool ratio-of-sums, not the mean of
    # jackknife pseudo-values.
    np.testing.assert_array_equal(
        result.signed_lit,
        np.concatenate([full_a, full_b], axis=1),
    )
    assert not np.allclose(result.signed_lit, np.mean(combined_blocks, axis=-1))
    np.testing.assert_array_equal(result.block_estimates, combined_blocks)
    np.testing.assert_allclose(result.statistical_covariance, expected_statistical)
    np.testing.assert_array_equal(result.systematic_error, expected_systematic)
    np.testing.assert_allclose(result.covariance, expected_covariance)
    assert np.any(np.abs(result.covariance[:, :2, 2:]) > 0.0)
    assert result.axes == "xz"
    np.testing.assert_array_equal(result.axis_indices, [0, 2])
    assert result.covariance_mode == "matched_blocks"
    assert result.metadata[0].observation_start == 0
    assert result.metadata[0].observation_stop == 2
    assert result.metadata[1].observation_start == 2
    assert result.metadata[1].observation_stop == 5
    assert result.metadata[1].eta_values == (0.08,)


def test_legacy_raw_blocks_are_rejected_even_with_jackknife(tmp_path: Path):
    path = tmp_path / "both-fields.npz"
    omega = np.asarray([0.1, 0.2])
    signed_lit = np.ones((2, 2))
    jackknife = np.arange(16, dtype=float).reshape(2, 2, 4)
    raw = jackknife + 1000.0
    _write_workflow_npz(
        path,
        omega=omega,
        eta=0.04,
        signed_lit=signed_lit,
        blocks=jackknife,
        raw_blocks=raw,
        digests="common-pool",
    )

    with pytest.raises(ValueError, match="unsupported legacy raw-block fields"):
        aggregate_lit_npz(path)


def test_mismatched_pools_fail_unless_independence_is_explicit(tmp_path: Path):
    omega = np.asarray([0.1, 0.2])
    signed_lit = np.ones((2, 2))
    blocks = np.arange(16, dtype=float).reshape(2, 2, 4)
    covariance_a = np.stack([np.eye(2), 2.0 * np.eye(2)])
    covariance_b = np.stack([3.0 * np.eye(2), 4.0 * np.eye(2)])
    path_a = tmp_path / "pool-a.npz"
    path_b = tmp_path / "pool-b.npz"
    _write_workflow_npz(
        path_a,
        omega=omega,
        eta=0.03,
        signed_lit=signed_lit,
        blocks=blocks,
        digests=("pool-a-x", "pool-a-z"),
        covariance=covariance_a,
    )
    _write_workflow_npz(
        path_b,
        omega=omega,
        eta=0.08,
        signed_lit=2.0 * signed_lit,
        blocks=blocks + 1.0,
        digests=("pool-b-x", "pool-b-z"),
        covariance=covariance_b,
    )

    with pytest.raises(ValueError, match="eval_pool_sha256 differs"):
        aggregate_lit_npz([path_a, path_b])

    result = aggregate_lit_npz(
        [path_a, path_b],
        assume_independent=True,
    )

    expected = np.zeros((2, 4, 4))
    expected[:, :2, :2] = covariance_a
    expected[:, 2:, 2:] = covariance_b
    np.testing.assert_array_equal(result.covariance, expected)
    np.testing.assert_array_equal(result.statistical_covariance, expected)
    assert result.block_estimates is None
    assert result.covariance_mode == "independent_files"
