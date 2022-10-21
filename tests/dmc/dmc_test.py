# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import jax.test_util as jtu
import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.dmc.dmc import do_run_dmc_single_walker, get_to_pad_num

class DmcTest(jtu.JaxTestCase):

    @staticmethod
    def make_dummy_func(expected_return_val):
        def dummy_func(position):
            return expected_return_val
        return dummy_func

    def test_single_walker_across_node(self):
        position = np.array([1.0, 0.0, 0.0])
        walker_age = 0.0

        def dummy_wave_func(position):
            if position[0] == 1.0:
                return (1, 1.0)
            else:
                return (-1, 1.0)

        def dummy_energy_func(position):
            if position[0] == 1.0:
                return 0.1
            else:
                return -0.1

        dummy_velocity = self.make_dummy_func(jnp.ones(3) * 0.1)
        dummy_local_energy = 0.0

        position_new, average_local_energy_new, walker_age_new, local_energy_new, weight_delta_log, delta_R, acceptance_rate, debug_info = do_run_dmc_single_walker(
            position,
            walker_age,
            dummy_local_energy,
            dummy_wave_func,
            dummy_velocity,
            dummy_energy_func,
            time_step=0.01,
            key=jax.random.PRNGKey(42),
            energy_offset=0.1,
            mixed_estimator=0.1,
            nuclei=jnp.ones((1, 3)),
            charges=jnp.ones(3))
        is_accepted, *_ = debug_info
        self.assertArraysEqual(position, position_new)
        self.assertEqual(average_local_energy_new, 0.0)
        self.assertEqual(walker_age_new, walker_age + 1)
        self.assertEqual(is_accepted, False)
        self.assertEqual(acceptance_rate, 0.0)

    def test_single_walker_should_accept_diffusion(self):
        position = np.array([1.0, 0.0, 0.0])
        walker_age = 0.0

        def dummy_wave_func(position):
            if position[0] == 1.0:
                return (1, 1.0)
            else:
                return (1, 1.0)

        def dummy_energy_func(position):
            if position[0] == 1.0:
                return 0.1
            else:
                return -0.1

        dummy_velocity = jnp.ones(3) * 0.1
        dummy_velocity_func = self.make_dummy_func(dummy_velocity)
        time_step=0.01
        dummy_local_energy = 0.0

        position_new, average_local_energy_new, walker_age_new, local_energy_new, weight_delta_log, delta_R, acceptance_rate, debug_info = do_run_dmc_single_walker(
            position,
            walker_age,
            dummy_local_energy,
            dummy_wave_func,
            dummy_velocity_func,
            dummy_energy_func,
            time_step=time_step,
            key=jax.random.PRNGKey(42),
            energy_offset=0.1,
            mixed_estimator=0.1,
            nuclei=jnp.ones((1, 3)),
            charges=jnp.ones(3))
        is_accepted, *_ = debug_info
        expected_energy_new = acceptance_rate * (-0.1) + (1 - acceptance_rate) * 0.1
        delta_position_norm = jnp.linalg.norm(
            position_new - position - dummy_velocity * time_step)

        self.assertEqual(is_accepted, True)
        self.assertNotAlmostEqual(position[0], position_new[0])
        self.assertLess(delta_position_norm, 2 * jnp.sqrt(time_step))
        self.assertEqual(average_local_energy_new, expected_energy_new)

    def test_single_walker_should_accept_old_walker(self):
        position = np.array([1.0, 0.0, 0.0])
        walker_age = 0.0
        walker_age_old = 100.0

        def dummy_wave_func(position):
            if position[0] == 1.0:
                return (1, 1.0)
            else:
                return (1, 0.1)

        def dummy_energy_func(position):
            if position[0] == 1.0:
                return 0.1
            else:
                return 0.1

        dummy_velocity = jnp.ones(3) * 0.1
        dummy_velocity_func = self.make_dummy_func(dummy_velocity)
        time_step=0.01
        dummy_local_energy = 0.0

        *_, debug_info = do_run_dmc_single_walker(
            position,
            walker_age,
            dummy_local_energy,
            dummy_wave_func,
            dummy_velocity_func,
            dummy_energy_func,
            time_step=time_step,
            key=jax.random.PRNGKey(42),
            energy_offset=0.1,
            mixed_estimator=0.1,
            nuclei=jnp.ones((1, 3)),
            charges=jnp.ones(3))
        is_accepted, acceptance_rate, *_ = debug_info
        self.assertEqual(is_accepted, False)

        position_new, average_local_energy_new, walker_age_new, local_energy_new, weight_delta_log, delta_R, acceptance_rate, debug_info = do_run_dmc_single_walker(
            position,
            walker_age_old,
            dummy_local_energy,
            dummy_wave_func,
            dummy_velocity_func,
            dummy_energy_func,
            time_step=time_step,
            key=jax.random.PRNGKey(42),
            energy_offset=0.1,
            mixed_estimator=0.1,
            nuclei=jnp.ones((1, 3)),
            charges=jnp.ones(3))

        is_accepted_old, *_ = debug_info
        delta_position_norm = jnp.linalg.norm(
            position_new - position - dummy_velocity * time_step)

        self.assertEqual(is_accepted_old, True)
        self.assertNotAlmostEqual(position[0], position_new[0])
        self.assertLess(delta_position_norm, 2 * jnp.sqrt(time_step))
        self.assertGreater(acceptance_rate, 0.8)

    def test_get_to_pad_num(self):
        data = [
            (4096, 8, 4160),
            (4095, 8, 4160),
            (4072, 8, 4160),
            (4071, 8, 4080),
            (40960, 8, 41600),
            (40920, 8, 41600),
            (40961, 8, 41600),
            (40800, 8, 41600),
            (40799, 8, 41600),
            (40791, 8, 40800),
        ]
        for num_walkers, num_device, expected_target_num in data:
            expected_to_pad_num = expected_target_num - num_walkers
            actual_result = get_to_pad_num(num_walkers, num_device)
            self.assertEqual(actual_result, expected_to_pad_num)

if __name__ == '__main__':
  absltest.main()

