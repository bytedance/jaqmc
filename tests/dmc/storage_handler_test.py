# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import chex

from jaqmc.dmc.storage_handler import local_storage_handler

class LocalStorageHandlerTest(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmpdir_path = pathlib.Path(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_rm_file(self):
        tmp_file = self.tmpdir_path / 'test.txt'
        tmp_file.touch()
        self.assertTrue(tmp_file.exists())

        local_storage_handler.rm(str(tmp_file))
        self.assertFalse(tmp_file.exists())

    def test_rm_dir(self):
        tmp_dir = self.tmpdir_path / 'test_dir/'
        tmp_dir.mkdir()
        self.assertTrue(tmp_dir.exists() and os.path.isdir(tmp_dir))

        local_storage_handler.rm(str(tmp_dir))
        self.assertFalse(tmp_dir.exists())

    @parameterized.parameters(True, False)
    def test_ls_path(self, return_fullpath):
        all_file_paths = []
        for i in range(6):
            tmp_filename = f'test_{i}.txt'
            tmp_file = self.tmpdir_path / tmp_filename
            tmp_file.touch()
            all_file_paths.append(str(tmp_file))
        expected_set = (set(all_file_paths)
                        if return_fullpath
                        else set(os.path.basename(f) for f in all_file_paths))

        ls_results = local_storage_handler.ls(self.tmpdir, return_fullpath=return_fullpath)
        self.assertSetEqual(set(ls_results), expected_set)

    def test_exists_no_file(self):
        fake_path = str(self.tmpdir_path / 'fake.txt')
        self.assertFalse(local_storage_handler.exists(fake_path))

    def test_exists(self):
        test_file = self.tmpdir_path / 'test.txt'
        test_file.touch()
        test_dir = self.tmpdir_path / 'test_dir/'
        test_dir.mkdir()

        self.assertTrue(local_storage_handler.exists(str(test_file)))
        self.assertTrue(local_storage_handler.exists(str(test_dir)))

    def test_exists_dir_no_dir(self):
        test_file = self.tmpdir_path / 'test.txt'
        test_file.touch()
        fake_path = str(self.tmpdir_path / 'fake/')

        self.assertFalse(local_storage_handler.exists_dir(str(test_file)))
        self.assertFalse(local_storage_handler.exists_dir(fake_path))

    def test_exists_dir(self):
        test_dir = self.tmpdir_path / 'test_dir/'
        test_dir.mkdir()

        self.assertTrue(local_storage_handler.exists_dir(str(test_dir)))

if __name__ == '__main__':
  absltest.main()
