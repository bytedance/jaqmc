# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import shutil
import subprocess
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from absl import app
from absl import flags
import chex

from jaqmc.dmc.storage_handler import HdfsStorageHandler

FLAGS = flags.FLAGS
flags.DEFINE_string('command_prefix', '', 'NA')

class HdfsStorageHandlerTest(chex.TestCase, parameterized.TestCase):
    def setUp(self):
        self.tmpdir = '__tmpdir'
        self.tmpdir_path = pathlib.Path('__tmpdir')
        self.command_prefix = FLAGS.command_prefix
        if self.command_prefix:
            subprocess.check_call([self.command_prefix, 'hdfs', 'dfs', '-mkdir', self.tmpdir])
        else:
            subprocess.check_call(['hdfs', 'dfs', '-mkdir', self.tmpdir])

        self.handler = HdfsStorageHandler(self.command_prefix)

        self.tmp_local_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self.command_prefix:
            subprocess.check_call([self.command_prefix, 'hdfs', 'dfs', '-rm', '-r', self.tmpdir])
        else:
            subprocess.check_call(['hdfs', 'dfs', '-rm', '-r', self.tmpdir])

        shutil.rmtree(self.tmp_local_dir)

    def make_temp_remote_directory(self):
        directory_path = os.path.join(self.tmpdir, 'test_dir')
        if self.command_prefix:
            subprocess.check_call([self.command_prefix, 'hdfs', 'dfs', '-mkdir', directory_path])
        else:
            subprocess.check_call(['hdfs', 'dfs', '-mkdir', directory_path])
        return directory_path

    def touch_temp_remote_file(self, filename='test.txt'):
        tmp_file = pathlib.Path(self.tmp_local_dir) / filename
        tmp_file.touch()
        remote_path = os.path.join(self.tmpdir, filename)
        if self.command_prefix:
            subprocess.check_call([self.command_prefix, 'hdfs', 'dfs', '-put', str(tmp_file), remote_path])
        else:
            subprocess.check_call(['hdfs', 'dfs', '-put', str(tmp_file), remote_path])
        return remote_path

    def test_rm_file(self):
        tmp_file = self.touch_temp_remote_file()
        self.assertTrue(self.handler.exists(tmp_file))

        self.handler.rm(tmp_file)
        self.assertFalse(self.handler.exists(tmp_file))

    def test_rm_dir(self):
        tmp_dir = self.make_temp_remote_directory()
        self.assertTrue(self.handler.exists_dir(tmp_dir))

        self.handler.rm(str(tmp_dir))
        self.assertFalse(self.handler.exists(tmp_dir))

    @parameterized.parameters(True, False)
    def test_ls_path(self, return_fullpath):
        all_file_paths = []
        for i in range(3):
            tmp_filename = f'test_{i}.txt'
            remote_path = self.touch_temp_remote_file(tmp_filename)
            all_file_paths.append(remote_path)

        remote_dir = self.make_temp_remote_directory()
        all_file_paths.append(remote_dir)

        expected_set = (set(all_file_paths)
                        if return_fullpath
                        else set(os.path.basename(f) for f in all_file_paths))

        ls_results = self.handler.ls(self.tmpdir, return_fullpath=return_fullpath)
        print('ls: ', ls_results)
        self.assertSetEqual(set(ls_results), expected_set)

    def test_exists_no_file(self):
        fake_path = str(self.tmpdir_path / 'fake.txt')
        self.assertFalse(self.handler.exists(fake_path))

    def test_exists(self):
        test_file = self.touch_temp_remote_file()
        test_dir = self.make_temp_remote_directory()

        self.assertTrue(self.handler.exists(test_file))
        self.assertTrue(self.handler.exists(test_dir))

    def test_exists_dir_no_dir(self):
        test_file = self.touch_temp_remote_file()
        fake_path = str(self.tmpdir_path / 'fake/')

        self.assertFalse(self.handler.exists_dir(test_file))
        self.assertFalse(self.handler.exists_dir(fake_path))

    def test_exists_dir(self):
        test_dir = self.make_temp_remote_directory()

        self.assertTrue(self.handler.exists_dir(test_dir))

if __name__ == '__main__':
  absltest.main()
