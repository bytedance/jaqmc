# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile

from absl.testing import absltest
import chex
import jax.numpy as jnp

from jaqmc.dmc.state import State, MixedEstimatorCalculator, EffectiveTimeStepCalculator
from jaqmc.dmc.ckpt_handler import CkptHandler
from jaqmc.dmc.ckpt_metric_manager import CkptMetricManager


class StateTest(chex.TestCase):

    def test_default_state(self):
        init_position = jnp.arange(12).reshape((6, 2))
        test_energy = -88.88
        def calc_energy_func(self, *args, **kwargs):
            return test_energy

        mixed_estimator_num_steps = 88
        energy_window_size = 22
        time_step = 1e-3
        init_step = 66

        state = State.default(
            init_position=init_position,
            calc_energy_func=calc_energy_func,
            mixed_estimator_num_steps=mixed_estimator_num_steps,
            energy_window_size=energy_window_size,
            time_step=time_step)

        chex.assert_tree_all_close(state.position, init_position)
        chex.assert_tree_all_close(state.walker_age, jnp.ones(len(init_position)))
        chex.assert_tree_all_close(state.weight, jnp.ones(len(init_position)))
        self.assertIsNone(state.local_energy)
        self.assertEqual(state.energy_offset, test_energy)
        self.assertEqual(state.target_num_walkers, len(init_position))
        self.assertEqual(state.mixed_estimator, test_energy)
        self.assertEqual(state.mixed_estimator_calculator.mixed_estimator_num_steps,
                         mixed_estimator_num_steps)
        self.assertEqual(state.mixed_estimator_calculator.all_energy.maxlen,
                         energy_window_size)
        self.assertEqual(state.effective_time_step_calculator.time_step, time_step)

if __name__ == '__main__':
  absltest.main()
