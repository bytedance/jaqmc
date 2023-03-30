import time
import unittest

import chex
import jax
import jax.numpy as jnp

from jaqmc.dmc.branch_fix_size import branch

class BranchFixSizeTest(chex.TestCase):
    def test_branch(self):
        # The first walker is 0-weight and therefore it's expected to be discarded.
        test_weight = jnp.array([0.0, 0.1, 0.5, 0.5, 4.0])
        key = jax.random.PRNGKey(int(1e6 * time.time()))
        before_branch_array = jnp.array([1, 2, 3, 4, 5])

        expected_after_branch_array = jnp.array([5, 2, 3, 4, 5])
        expected_weight = jnp.array([2.0, 0.1, 0.5, 0.5, 2.0])

        for test_min_thres, test_max_thres in [(0.3, 2.0),
                                               (-1.0, 2.0),
                                               (0.2, 5.0)]:

            with self.subTest(msg=f'check min_thres {test_min_thres} and max_thres {test_max_thres}'):
                actual_weight, [actual_branch_array] = branch(test_weight, key, [before_branch_array],
                                                              min_thres=test_min_thres, max_thres=test_max_thres)
                self.assertSequenceAlmostEqual(expected_weight, actual_weight)
                self.assertSequenceAlmostEqual(expected_after_branch_array, actual_branch_array)

if __name__ == '__main__':
    unittest.main()
