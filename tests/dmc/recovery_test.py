# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import tempfile
import shutil

from jaqmc.dmc.ckpt_metric_manager import CkptMetricManager, NoSafeDataAvailable
from jaqmc.dmc.dmc import recovery_wrapper, IterationOutput
from jaqmc.dmc.state import State

class TestException(Exception):
    pass

class RecoveryWrapperTest(chex.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    _test_iteration_output = IterationOutput(
        succeeded=True,
        state=State(position=jnp.arange(20).reshape((10, 2))),
        key=None)
    _test_step = 666
    def _test_dmc_iteration(self, should_raise):
        if should_raise:
            raise TestException()
        return self._test_step, self._test_iteration_output

    def test_no_failure(self):
        # When no failure happens, the raw function's behavior should not
        # be modified at all
        wrapped_func = recovery_wrapper(self._test_dmc_iteration, 0,
                                        ckpt_metric_manager=None,
                                        key=None)
        step, output = wrapped_func(should_raise=False)
        self.assertEqual(step, self._test_step)
        chex.assert_trees_all_close(output.state.position,
                                    self._test_iteration_output.state.position)

    def test_no_safe_data_available(self):
        ckpt_metric_manager = CkptMetricManager(metric_schema=[], block_size=3,
                                                lazy_setup=False)

        wrapped_func = recovery_wrapper(self._test_dmc_iteration,
                                        max_restore_nums=1,
                                        ckpt_metric_manager=ckpt_metric_manager,
                                        key=None)
        self.assertRaises(NoSafeDataAvailable, wrapped_func, should_raise=True)

    def get_manager_with_safe_data(self, block_size=10):
        position = jnp.arange(120).reshape((10, 12))
        state = State(position=position)

        ckpt_metric_manager = CkptMetricManager(metric_schema=[],
                                                block_size=block_size,
                                                save_path=self.tmpdir,
                                                lazy_setup=False)
        with ckpt_metric_manager:
            for i in range(block_size + 1):
                ckpt_metric_manager.run(i, state, [])
        return ckpt_metric_manager, state

    def test_has_safe_data_available(self):
        max_restore_nums = 3

        counter = 0
        def test_func():
            counter += 1
            if counter == 1:
                raise Exception()
            return counter, None

        def get_metric_latest_step(ckpt_metric_manager):
            data = ckpt_metric_manager.alive_metric_manager.get_metric_data()
            return data.step.iloc[-1]

        block_size = 10
        ckpt_metric_manager, state = self.get_manager_with_safe_data(block_size)
        self.assertEqual(get_metric_latest_step(ckpt_metric_manager), block_size)

        key = jax.random.PRNGKey(666)
        wrapped_func = recovery_wrapper(test_func,
                                        max_restore_nums=max_restore_nums,
                                        ckpt_metric_manager=ckpt_metric_manager,
                                        key=key)
        step, output = wrapped_func()
        self.assertEqual(step, 1)
        self.assertFalse(output.succeeded)
        chex.assert_trees_all_close(output.state.position, state.position)

        # metric should also be reverted
        self.assertEqual(get_metric_latest_step(ckpt_metric_manager), 0)

    def test_has_safe_data_available_but_exceeds_max_num(self):
        max_restore_nums = 3

        def test_func():
            raise TestException()
        ckpt_metric_manager, state = self.get_manager_with_safe_data()
        key = jax.random.PRNGKey(666)
        wrapped_func = recovery_wrapper(test_func,
                                        max_restore_nums=max_restore_nums,
                                        ckpt_metric_manager=ckpt_metric_manager,
                                        key=key)
        # First `max_restore_nums` exception will be captured and retry be activated.
        for _ in range(max_restore_nums):
            step, output = wrapped_func()
            self.assertFalse(output.succeeded)

        self.assertRaises(TestException, wrapped_func)

if __name__ == '__main__':
  absltest.main()
