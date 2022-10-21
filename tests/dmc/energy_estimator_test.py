# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque

from absl.testing import absltest
import jax.test_util as jtu
import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.dmc.energy_estimator import MixedEstimatorCalculator

def get_mixed_estimator_Pi(energy_offsets, time_step):
    log = -time_step * np.sum(np.array(energy_offsets))
    return np.exp(log)

class EnergyEstimatorTest(jtu.JaxTestCase):

    @staticmethod
    def calc_estimator(all_energy_offsets, all_energy, all_total_weight,
                       mixed_estimator_num_steps, energy_window_size, all_time_step):
        all_Pi = []
        for i in range(len(all_energy_offsets)):
            start = max(0, i + 1 - mixed_estimator_num_steps)
            Pi = get_mixed_estimator_Pi(all_energy_offsets[start: (i + 1)], all_time_step[i])
            all_Pi.append(Pi)
        numerator = 0
        denominator = 0
        start_index = len(all_energy) - energy_window_size
        for i, (energy, total_weight, Pi) in enumerate(zip(all_energy, all_total_weight, all_Pi)):
            if i < start_index:
                continue
            numerator += Pi * energy * total_weight
            denominator += Pi * total_weight
        return numerator / denominator

    def test_mixed_estimator_Pi(self):
        all_energy_offsets = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        all_energy = [1.1, 2.2, 3.5, 1.0, 2.2, 3.0]
        all_total_weight = [1.0, 2, 1.0, 1.0, 1.0, 1.0]
        all_time_step = [1e-2, 1e-1, 1e-1, 1e-2, 1e-3]

        mixed_estimator_num_steps = 2
        energy_window_size = 6
        calculator = MixedEstimatorCalculator(mixed_estimator_num_steps,
                                              energy_window_size)
        for i, (energy_offset, energy, total_weight, time_step) in enumerate(zip(all_energy_offsets, all_energy, all_total_weight, all_time_step)):
            expected_result = self.calc_estimator(
                all_energy_offsets[:(i + 1)],
                all_energy[:(i + 1)],
                all_total_weight[:(i + 1)],
                mixed_estimator_num_steps=mixed_estimator_num_steps,
                energy_window_size=energy_window_size,
                all_time_step=all_time_step)
            result = calculator.run(energy_offset, energy, total_weight, time_step)
            self.assertAlmostEqual(result, expected_result, places=5)

    def test_mixed_estimator_Pi_2(self):
        all_energy_offsets = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        all_energy = [1.1, 2.2, 3.5, 1.0, 2.2, 3.0]
        all_total_weight = [1.0, 2, 1.0, 1.0, 1.0, 1.0]
        all_time_step = [1e-2, 1e-1, 1e-1, 1e-2, 1e-3]

        mixed_estimator_num_steps = 2
        energy_window_size = 3
        calculator = MixedEstimatorCalculator(mixed_estimator_num_steps,
                                              energy_window_size)
        for i, (energy_offset, energy, total_weight, time_step) in enumerate(zip(all_energy_offsets, all_energy, all_total_weight, all_time_step)):
            expected_result = self.calc_estimator(
                all_energy_offsets[:(i + 1)],
                all_energy[:(i + 1)],
                all_total_weight[:(i + 1)],
                mixed_estimator_num_steps=mixed_estimator_num_steps,
                energy_window_size=energy_window_size,
                all_time_step=all_time_step)
            result = calculator.run(energy_offset, energy, total_weight, time_step=time_step)
            self.assertAlmostEqual(result, expected_result, places=5)

    def test_mixed_estimator_Pi_3(self):
        all_energy_offsets = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        all_energy = [1.1, 2.2, 3.5, 1.0, 2.2, 3.0]
        all_total_weight = [1.0, 2, 1.0, 1.0, 1.0, 1.0]
        all_time_step = [1e-2, 1e-1, 1e-1, 1e-2, 1e-3]

        mixed_estimator_num_steps = 4
        energy_window_size = 3
        calculator = MixedEstimatorCalculator(mixed_estimator_num_steps,
                                              energy_window_size)
        for i, (energy_offset, energy, total_weight, time_step) in enumerate(zip(all_energy_offsets, all_energy, all_total_weight, all_time_step)):
            expected_result = self.calc_estimator(
                all_energy_offsets[:(i + 1)],
                all_energy[:(i + 1)],
                all_total_weight[:(i + 1)],
                mixed_estimator_num_steps=mixed_estimator_num_steps,
                energy_window_size=energy_window_size,
                all_time_step=all_time_step)
            result = calculator.run(energy_offset, energy, total_weight, time_step=time_step)
            self.assertAlmostEqual(result, expected_result, places=5)

if __name__ == '__main__':
  absltest.main()
