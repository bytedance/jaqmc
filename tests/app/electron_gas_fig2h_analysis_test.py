# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Focused test for the Fig. 2h evaluation analysis."""

import h5py
import numpy as np
import pytest

from jaqmc.app.electron_gas.fig2h_analysis import (
    N_ELECTRONS,
    analyze_evaluation,
)


def test_reblocks_per_electron_energy_and_uses_paper_formula(tmp_path) -> None:
    rng = np.random.default_rng(0)
    innovations = rng.normal(scale=0.02, size=16_384)
    correlated = np.empty_like(innovations)
    correlated[0] = innovations[0]
    for index in range(1, correlated.size):
        correlated[index] = 0.9 * correlated[index - 1] + innovations[index]

    paper_energy = 0.530019
    path = tmp_path / "evaluation_stats.h5"
    with h5py.File(path, "w") as handle:
        handle["total_energy"] = N_ELECTRONS * (
            paper_energy + 1e-3 * correlated + 2e-6j
        )

    result = analyze_evaluation(
        path,
        {
            "rs": 1.0,
            "hartree_fock": 0.5689,
            "bf_dmc": 0.52989,
            "net": paper_energy,
            "bf_vmc": 0.53009,
            "tc_dcd": 0.52968,
            "dcd_supplement_table_14": 0.53001,
            "tc_fciqmc": 0.52973,
        },
        expected_steps=correlated.size,
    )

    assert result["energy_per_electron"] == pytest.approx(
        paper_energy + 1e-3 * correlated.mean()
    )
    assert result["standard_error_per_electron"] > 0
    assert result["reblocking"]["level"] > 0
    assert result["imaginary_energy_mean_per_electron"] == pytest.approx(2e-6)
    assert result["imaginary_energy_max_abs_per_electron"] == pytest.approx(2e-6)
    assert result["paper_net_correlation_error_percent"] == pytest.approx(0.3306844399)
