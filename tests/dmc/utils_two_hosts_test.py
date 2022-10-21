# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import jax.test_util as jtu
import jax.numpy as jnp

from jaqmc.dmc.utils import agg_sum, agg_mean
from jaqmc.dmc.runner import initialize_distributed_runtime

NUM_HOST = 2

class UtilsTwoHostsTest(jtu.JaxTestCase):
    def test_agg_mean(self):
        x = jnp.arange(1, 10)
        actual_mean = agg_mean(x)

        self.assertEqual(actual_mean, 5.0)

    def test_agg_mean_weighted(self):
        x = jnp.arange(1, 10)
        weights = jnp.array([1.0] * 5 + [0.0] * 4)
        actual_mean = agg_mean(x, weights=weights)

        self.assertEqual(actual_mean, 3)

    def test_agg_sum(self):
        x = jnp.arange(1, 101)
        actual_sum = agg_sum(x)

        single_host_result = 5050
        self.assertEqual(actual_sum, single_host_result * NUM_HOST)

if __name__ == '__main__':
  initialize_distributed_runtime()
  absltest.main()
