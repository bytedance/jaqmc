# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from absl.testing import absltest
import chex
import tempfile
import shutil

from jaqmc.dmc.data_path import DataPath, _resolve_path, _setup_path
from jaqmc.dmc.storage_handler import dummy_storage_handler, local_storage_handler

class DataPathTest(chex.TestCase):

    test_local_path = 'test/local/path'
    test_remote_path = 'test/remote/path'
    test_remote_path2 = 'test/remote/path2'

    def test_resolve_string_path(self):
        path = _resolve_path(self.test_local_path)
        self.assertIsInstance(path, DataPath)
        self.assertEqual(path.local_path, self.test_local_path)
        self.assertIsNone(path.remote_path)


    def test_setup_local_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = _resolve_path(os.path.join(tmpdir, self.test_local_path))
            restore_path = DataPath()
            updated_save_path, updated_restore_path = _setup_path(
                save_path=save_path,
                restore_path=restore_path,
                remote_storage_handler=dummy_storage_handler)

            self.assertEqual(updated_save_path.local_path, save_path.local_path)
            self.assertEqual(updated_save_path.remote_path, save_path.remote_path)
            self.assertEqual(updated_restore_path.remote_path, restore_path.remote_path)
            # The empty local path should be replaced by some temp path
            self.assertNotEqual(updated_restore_path.local_path, restore_path.local_path)

            self.assertTrue(local_storage_handler.exists_dir(updated_save_path.local_path))
            self.assertTrue(local_storage_handler.exists_dir(updated_restore_path.local_path))

        # Remove the temp path for clean-up
        shutil.rmtree(updated_restore_path.local_path)

    def test_setup_remote_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.TemporaryDirectory() as tmp_remote_dir:
                save_path = DataPath(
                    local_path=os.path.join(tmpdir, self.test_local_path),
                    remote_path=os.path.join(tmp_remote_dir, self.test_remote_path))
                restore_path = DataPath(remote_path=os.path.join(tmpdir, self.test_remote_path2))
                updated_save_path, updated_restore_path = _setup_path(
                    save_path=save_path,
                    restore_path=restore_path,
                    remote_storage_handler=local_storage_handler)

                self.assertEqual(updated_save_path.local_path, save_path.local_path)
                self.assertEqual(updated_save_path.remote_path, save_path.remote_path)
                self.assertEqual(updated_restore_path.remote_path, restore_path.remote_path)
                # The empty local path should be replaced by some temp path
                self.assertNotEqual(updated_restore_path.local_path, restore_path.local_path)

                self.assertTrue(local_storage_handler.exists_dir(updated_save_path.local_path))
                self.assertTrue(local_storage_handler.exists_dir(updated_save_path.remote_path))
                self.assertTrue(local_storage_handler.exists_dir(updated_restore_path.local_path))

        # Remove the temp path for clean-up
        shutil.rmtree(updated_restore_path.local_path)

if __name__ == '__main__':
  absltest.main()
