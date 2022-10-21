# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import jax.test_util as jtu
import jax
import jax.numpy as jnp

from jaqmc.dmc.branch import branch, do_branch, round_merge_pairs

class BranchTest(jtu.JaxTestCase):
    def test_do_branch(self):
        key = jax.random.PRNGKey(42)
        min_thres = 0.5
        max_thres = 2
        weight = jnp.array([0.01, 0.02, 0.1, 0.1, 0.2, 0.3, 0.44, 1.6, 1.8, 3, 4])

        updated_weight, repeat_num = do_branch(
            weight,
            key,
            5,
            min_thres=min_thres,
            max_thres=max_thres)

        def total_weight(weight, repeat_num):
            return jnp.sum(weight * repeat_num)

        expected_weight = jnp.array([0.01, 0.03, 0.2 , 0.1 , 0.5 , 0.3 , 0.44, 1.02, 1.8 , 1.5 , 2])
        expected_repeat_num = jnp.array([0, 1, 1, 0, 1, 0, 0, 2, 1, 2, 2])

        self.assertAlmostEqual(total_weight(updated_weight, repeat_num), jnp.sum(weight), places=5)
        self.assertArraysEqual(updated_weight, expected_weight)
        self.assertArraysEqual(repeat_num, expected_repeat_num)

    def test_round_merge_pairs(self):
        for num in range(11):
            self.assertEqual(round_merge_pairs(num), num)

        for num, expected_round_result in [
                (25, 20),
                (35, 30),
                (120, 100),
                (1200, 1000),
                (2200, 2000),
                (9200, 9000)]:
            self.assertEqual(round_merge_pairs(num), expected_round_result)

if __name__ == '__main__':
  absltest.main()
