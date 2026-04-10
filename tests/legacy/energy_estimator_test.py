# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from jaqmc_legacy.dmc.energy_estimator import MixedEstimatorCalculator


def _mixed_estimator_pi(energy_offsets: list[float], time_step: float) -> float:
    log = -time_step * np.sum(np.array(energy_offsets))
    return float(np.exp(log))


def _calc_mixed_estimator(
    all_energy_offsets: list[float],
    all_energy: list[float],
    all_total_weight: list[float],
    *,
    mixed_estimator_num_steps: int,
    energy_window_size: int,
    all_time_step: list[float],
) -> float:
    all_pi: list[float] = []
    for i in range(len(all_energy_offsets)):
        start = max(0, i + 1 - mixed_estimator_num_steps)
        pi = _mixed_estimator_pi(all_energy_offsets[start : (i + 1)], all_time_step[i])
        all_pi.append(pi)
    numerator = 0.0
    denominator = 0.0
    start_index = len(all_energy) - energy_window_size
    for i, (energy, total_weight, pi) in enumerate(
        zip(all_energy, all_total_weight, all_pi)
    ):
        if i < start_index:
            continue
        numerator += pi * energy * total_weight
        denominator += pi * total_weight
    return numerator / denominator


@pytest.mark.parametrize("energy_window_size", [6, 3])
@pytest.mark.parametrize("mixed_estimator_num_steps", [2, 4])
def test_mixed_estimator_pi(
    mixed_estimator_num_steps: int,
    energy_window_size: int,
) -> None:
    all_energy_offsets = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    all_energy = [1.1, 2.2, 3.5, 1.0, 2.2, 3.0]
    all_total_weight = [1.0, 2, 1.0, 1.0, 1.0, 1.0]
    all_time_step = [1e-2, 1e-1, 1e-1, 1e-2, 1e-3, 1e-3]

    calculator = MixedEstimatorCalculator(
        mixed_estimator_num_steps,
        energy_window_size,
    )
    for i, (energy_offset, energy, total_weight, time_step_val) in enumerate(
        zip(all_energy_offsets, all_energy, all_total_weight, all_time_step)
    ):
        expected_result = _calc_mixed_estimator(
            all_energy_offsets[: (i + 1)],
            all_energy[: (i + 1)],
            all_total_weight[: (i + 1)],
            mixed_estimator_num_steps=mixed_estimator_num_steps,
            energy_window_size=energy_window_size,
            all_time_step=all_time_step,
        )
        result = calculator.run(
            energy_offset,
            energy,
            total_weight,
            time_step=time_step_val,
        )
        assert float(result) == pytest.approx(float(expected_result), rel=1e-5)
