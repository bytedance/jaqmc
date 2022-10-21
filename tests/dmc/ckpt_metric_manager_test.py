# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import pathlib
import shutil
import tempfile
import tarfile

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import pandas as pd


from jaqmc.dmc.ckpt_metric_manager import CkptMetricManager, NoSafeDataAvailable
from jaqmc.dmc.data_path import DataPath
from jaqmc.dmc.storage_handler import local_storage_handler
from jaqmc.dmc.ckpt_handler import CkptHandler

class CkptMetricManagerTest(chex.TestCase, parameterized.TestCase):

    data_file_prefix = 'test_data'
    ckpt_file_prefix = 'test_ckpt'
    metric_file_name = 'text_metric'
    schema =['t', 'e', 's']

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def get_manager(self, block_size, **kwargs):
        return CkptMetricManager(
            metric_schema=self.schema,
            block_size=block_size,
            data_file_prefix=self.data_file_prefix,
            ckpt_file_prefix=self.ckpt_file_prefix,
            metric_file_name=self.metric_file_name,
            **kwargs)

    def test_lazy_initialization(self):
        # No actual file system stuff should be done at the construction time of
        # the CkptMetricManager.
        manager = self.get_manager(
            block_size=10,
            # These two line should cause issue if any actual file-system
            # action is performed.
            remote_storage_handler=None,
            local_storage_handler=None,
            lazy_setup=True)

        def enter_context():
            # Actually file-system action will be performed when entering
            # the context, at which time some exception will be raised due
            # to wrongly-passed-in storage handlers.
            with manager:
                pass
        self.assertRaises(Exception, enter_context)

    def test_always_use_brand_new_workspace(self):
        tmpdir = pathlib.Path(self.tmpdir)
        save_path = tmpdir / 'save'
        target_workspace_path = save_path / '__workspace'
        dummy_dir = target_workspace_path / '__dummy__'
        dummy_dir.mkdir(parents=True)

        self.assertTrue(dummy_dir.exists())

        manager = self.get_manager(save_path=str(save_path), block_size=10,
                                   lazy_setup=False)

        # Workspace will be created when doing setup
        self.assertFalse(dummy_dir.exists())

    def test_always_use_brand_new_workspace_lazy_setup(self):
        tmpdir = pathlib.Path(self.tmpdir)
        save_path = tmpdir / 'save'
        target_workspace_path = save_path / '__workspace'
        dummy_dir = target_workspace_path / '__dummy__'
        dummy_dir.mkdir(parents=True)


        manager = self.get_manager(save_path=str(save_path), block_size=10,
                                   lazy_setup=True)

        self.assertTrue(dummy_dir.exists())
        # Workspace will be created when doing setup.
        # When doing lazy-setup, it's done when entering the context.
        with manager:
            self.assertFalse(dummy_dir.exists())


    @parameterized.parameters(*[(3, i) for i in range(1, 11)])
    def test_upload_safe_data(self, block_size, N):
        tmpdir = pathlib.Path(self.tmpdir)
        local_path = tmpdir / 'local'
        remote_path = tmpdir / 'remote'
        save_path = DataPath(str(local_path), str(remote_path))

        manager = self.get_manager(
            save_path=save_path,
            block_size=block_size,
            remote_storage_handler=local_storage_handler)

        dummy_metric = [1, 2, 3]
        dummy_data = jnp.array([])
        with manager:
            for i in range(N):
                manager.run(i, dummy_data, dummy_metric)
        all_local_files = [f for f in local_storage_handler.ls(remote_path) if os.path.isfile(f)]
        all_remote_files = local_storage_handler.ls(remote_path)

        self.assertEqual(len(all_remote_files), (N - 1) // block_size)
        self.assertEqual(len(all_local_files), (N - 1) // block_size)

        # Local and remote save paths should have the same set of safe data files
        all_local_file_names = set([os.path.basename(f) for f in all_local_files])
        all_remote_file_names = set([os.path.basename(f) for f in all_remote_files])
        self.assertSetEqual(all_local_file_names, all_remote_file_names)


    @staticmethod
    def expected_safe_data_num(block_size, N, start_from_scratch):
        # Note that the latest ckpt is marked as unsafe, so we need to do
        # "minus one" on the calculated number of ckpts.

        # When starting from scratch, we will save ckpt at 0-th step.
        if start_from_scratch:
            if N <= block_size:
                return 0
            # We can simply do (N - 1) // block_size - 1, but it's trickier to reason about.
            if N % block_size == 0:
                return N // block_size - 1
            else:
                return (N // block_size + 1) - 1

        # When (re-)starting from a checkpoint, the first ckpt will have
        # index "latest ckpt index" + block_size.
        else:
            if N <= block_size:
                return 0
            else:
                return N // block_size - 1

    def list_data(self, directory, storage_handler, data_file_prefix=None):
        data_file_prefix = data_file_prefix or self.data_file_prefix
        all_paths = storage_handler.ls(directory)
        return [f for f in all_paths if os.path.isfile(f) and os.path.basename(f).startswith(data_file_prefix)]

    @parameterized.parameters(*[(3, i) for i in range(1, 11)])
    def test_safe_ckpt_and_metric_has_same_step(self, block_size, N):
        tmpdir = pathlib.Path(self.tmpdir)
        local_path = tmpdir / 'local'
        remote_path = tmpdir / 'remote'
        save_path = DataPath(str(local_path), str(remote_path))

        manager = self.get_manager(
            save_path=save_path,
            block_size=block_size,
            remote_storage_handler=local_storage_handler)

        dummy_metric = [1, 2, 3]
        dummy_data = jnp.array([])
        with manager:
            for i in range(N):
                manager.run(i, dummy_data, dummy_metric)
        all_local_files = self.list_data(local_path, local_storage_handler)
        all_remote_files = self.list_data(remote_path, local_storage_handler)

        def load_and_compare(path):
            extract_dir = tempfile.mkdtemp(dir=tmpdir)
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(extract_dir)

            # Our tarball should contain one data file and another metric file
            all_paths = local_storage_handler.ls(extract_dir)
            ckpt_paths = [f for f in all_paths if os.path.basename(f).startswith(self.ckpt_file_prefix)]
            assert len(ckpt_paths) == 1
            ckpt_path = ckpt_paths[0]

            assert os.path.join(extract_dir, self.metric_file_name) in all_paths

            ckpt_step, ckpt_data = CkptHandler._load_ckpt(ckpt_path)
            metric_df = pd.read_csv(os.path.join(extract_dir, self.metric_file_name))
            metric_step = metric_df.iloc[-1].step
            assert ckpt_step == metric_step

        for f in itertools.chain(all_local_files, all_remote_files):
            load_and_compare(f)

    @parameterized.parameters(*[(i, j, k) for i in range(1, 11) for j in range(1, 11) for k in range(1, 3)])
    def test_restore_and_rollback(self, N, M, block_size):
        '''
        We mainly care about two things:
        1. the ckpt is successfully restored
        2. the metric file is normal
        '''
        tmpdir = pathlib.Path(self.tmpdir)
        local_path = tmpdir / 'local'
        remote_path = tmpdir / 'remote'
        save_path = DataPath(str(local_path), str(remote_path))

        manager = self.get_manager(
            save_path=save_path,
            block_size=block_size,
            remote_storage_handler=local_storage_handler)

        dummy_metric = [1, 2, 3]
        dummy_data = jnp.array([])

        def load(path):
            extract_dir = tempfile.mkdtemp(dir=tmpdir)
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(extract_dir)

            # Our tarball should contain one data file and another metric file
            all_paths = local_storage_handler.ls(extract_dir)
            ckpt_paths = [f for f in all_paths if os.path.basename(f).startswith(self.ckpt_file_prefix)]
            assert len(ckpt_paths) == 1
            ckpt_path = ckpt_paths[0]

            assert os.path.join(extract_dir, self.metric_file_name) in all_paths

            ckpt_step, ckpt_data = CkptHandler._load_ckpt(ckpt_path)
            metric_df = pd.read_csv(os.path.join(extract_dir, self.metric_file_name))
            return metric_df, ckpt_step, ckpt_data

        with manager:
            for i in range(N):
                manager.run(i, dummy_data, dummy_metric)
            all_local_files_before_restore = self.list_data(local_path, local_storage_handler)
            all_remote_files_before_restore = self.list_data(remote_path, local_storage_handler)
            if (N - 1) // block_size < 1:
                self.assertEmpty(all_local_files_before_restore)
                self.assertEmpty(all_remote_files_before_restore)
                self.assertRaises(NoSafeDataAvailable, manager.restore_and_rollback)
            else:
                expected_data_num_before_restore = self.expected_safe_data_num(block_size, N, start_from_scratch=True)
                self.assertEqual(len(all_local_files_before_restore), expected_data_num_before_restore)
                self.assertEqual(len(all_remote_files_before_restore), expected_data_num_before_restore)

                ## Now the state should be rollback to ((N - 1) // block_size) * block_size
                step, ckpt_data = manager.restore_and_rollback()

                for i in range(step + 1, M + step + 1):
                    manager.run(i, dummy_data, dummy_metric)

                all_local_files = self.list_data(local_path, local_storage_handler)
                all_remote_files = self.list_data(remote_path, local_storage_handler)
                expected_data_num_after_restore = self.expected_safe_data_num(block_size, M, start_from_scratch=False)
                expected_total_data_num = expected_data_num_before_restore + expected_data_num_after_restore
                self.assertEqual(len(all_remote_files), expected_total_data_num)
                self.assertEqual(len(all_local_files), expected_total_data_num)
                for f in all_remote_files:
                    metric_df, ckpt_step, ckpt_data = load(f)
                    metric_df_indices = metric_df.step.tolist()
                    self.assertListEqual(metric_df_indices, list(range(metric_df_indices[-1] + 1)))

    @parameterized.parameters(*[(i, k) for k in range(1, 3) for i in range(k + 1, 11)])
    def test_restore_and_rollback_clear_unsafe_data(self, N, block_size):
        '''
        We mainly care about two things:
        1. the ckpt is successfully restored
        2. the metric file is normal
        '''
        tmpdir = pathlib.Path(self.tmpdir)
        local_path = tmpdir / 'local'
        remote_path = tmpdir / 'remote'
        save_path = DataPath(str(local_path), str(remote_path))

        manager = self.get_manager(
            save_path=save_path,
            block_size=block_size,
            remote_storage_handler=local_storage_handler)

        dummy_metric = [1, 2, 3]
        dummy_data = jnp.array([])
        with manager:
            for i in range(N):
                manager.run(i, dummy_data, dummy_metric)

            manager.restore_and_rollback()
            self.assertFalse(manager.has_unsafe_data)

    @parameterized.parameters(*[(i, k) for k in range(1, 3) for i in range(k + 1, 11) ])
    def test_save_unsafe_and_commit_safe_data_no_unsafe_data(self, N, block_size):
        tmpdir = pathlib.Path(self.tmpdir)
        local_path = tmpdir / 'local'
        remote_path = tmpdir / 'remote'
        save_path = DataPath(str(local_path), str(remote_path))

        manager = self.get_manager(
            save_path=save_path,
            block_size=block_size,
            remote_storage_handler=local_storage_handler)

        dummy_metric = [1, 2, 3]
        dummy_data = jnp.array([])

        with manager:
            for i in range(N):
                manager.run(i, dummy_data, dummy_metric)

            # Right after restoring, the unsafe data is erased.
            manager.restore_and_rollback()

            # Clear all data in save_path manually.
            all_local_files = self.list_data(local_path, local_storage_handler)
            all_remote_files = self.list_data(remote_path, local_storage_handler)
            for f in itertools.chain(all_local_files, all_remote_files):
                os.remove(f)

            manager._save_unsafe_and_commit_safe_data(N, dummy_data)

            all_local_files_after_commit = self.list_data(local_path, local_storage_handler)
            all_remote_files_after_commit = self.list_data(remote_path, local_storage_handler)
            self.assertEmpty(all_local_files_after_commit)
            self.assertEmpty(all_remote_files_after_commit)

    @parameterized.parameters(*[(i, k) for k in range(1, 3) for i in range(1, 11) ])
    def test_save_unsafe_and_commit_safe_data_has_unsafe_data(self, N, block_size):
        tmpdir = pathlib.Path(self.tmpdir)
        local_path = tmpdir / 'local'
        remote_path = tmpdir / 'remote'
        save_path = DataPath(str(local_path), str(remote_path))

        manager = self.get_manager(
            save_path=save_path,
            block_size=block_size,
            remote_storage_handler=local_storage_handler)

        dummy_metric = [1, 2, 3]
        dummy_data = jnp.array([])

        with manager:
            for i in range(N):
                manager.run(i, dummy_data, dummy_metric)

            unsafe_step = manager.unsafe_step

            # Clear all data in save_path manually.
            all_local_files = self.list_data(local_path, local_storage_handler)
            all_remote_files = self.list_data(remote_path, local_storage_handler)
            for f in itertools.chain(all_local_files, all_remote_files):
                os.remove(f)

            manager._save_unsafe_and_commit_safe_data(N, dummy_data)

            all_local_files_after_commit = self.list_data(local_path, local_storage_handler)
            all_remote_files_after_commit = self.list_data(remote_path, local_storage_handler)

            def assert_correct_step(f):
                data_index = manager._get_data_index(os.path.basename(f))
                self.assertEqual(unsafe_step, data_index)
            self.assertEqual(len(all_local_files_after_commit), 1)
            self.assertEqual(len(all_remote_files_after_commit), 1)
            assert_correct_step(all_local_files_after_commit[0])
            assert_correct_step(all_remote_files_after_commit[0])

if __name__ == '__main__':
  absltest.main()
