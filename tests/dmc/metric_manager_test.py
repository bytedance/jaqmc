# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile

from absl.testing import absltest
import chex
import numpy as np

from jaqmc.dmc.metric_manager import MetricManager

class MetricManagerTest(chex.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    test_filename = 'test.csv'

    def test_write(self):
        schema = ['a', 'b']
        metric_manager = MetricManager(self.test_filename, schema, self.tmpdir)

        test_data = np.arange(8).reshape(4, 2)
        with metric_manager:
            for i, data in enumerate(test_data):
                metric_manager.write(i, data)

            data = metric_manager.get_metric_data()

        target_a_col = test_data[:, 0]
        target_b_col = test_data[:, 1]
        chex.assert_trees_all_close(data.a.to_numpy(), target_a_col)
        chex.assert_trees_all_close(data.b.to_numpy(), target_b_col)

if __name__ == '__main__':
  absltest.main()
