# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Handling both ckpt and metric file saving / loading / rolling back
'''

from concurrent.futures import ThreadPoolExecutor
import itertools
import os
import re
import shutil
import tarfile
from typing import List, Optional, Union, Tuple

from absl import logging

from .ckpt_handler import CkptHandler
from .data_path import DataPath, initialize_data_paths
from .metric_manager import MetricManager
from .storage_handler import (StorageHandler,
                              dummy_storage_handler,
                              local_storage_handler)
from .state import State

class NoSafeDataAvailable(Exception):
    '''
    Exception raised when no safe data is available for restoring.
    '''

class NotSetupYet(Exception):
    '''
    Exception raised when no safe data is available for restoring.
    '''

class CkptMetricManager:
    '''
    A context manager managing both checkpoints (ckpt for short) and metric files.

    It supports save to and restoring from both local and remote paths.
    It also supports revert to the latest "safe" state, say, when hitting a recoverable
    failure.
    '''

    def __init__(self,
                 metric_schema: List[str],
                 block_size: int,
                 data_file_prefix: str = 'data',
                 ckpt_file_prefix: str = 'ckpt',
                 metric_file_name: str = 'metric.csv',
                 save_path: Optional[Union[DataPath, str]] = None,
                 restore_path: Optional[Union[DataPath, str]] = None,
                 remote_storage_handler: StorageHandler = dummy_storage_handler,
                 local_storage_handler: StorageHandler = local_storage_handler,
                 lazy_setup: bool = False):
        '''
        Args:
            metric_schema: The column names for the metric file.
            block_size: A block is a group of iterations. When (recoverable) error
                        happens, we will restart from a ckpt one block away.
                        We will save ckpt every `block_size` of iterations.
            data_file_prefix: The prefix (and the identifier) of "data" files.
                              A data file is contains all the data saved every
                              block. For now it's simply a tarball of one ckpt file
                              and one metric file.
            ckpt_file_prefix: The prefix (and the identifier) of "data" files.
                              The ckpt file contains all the (internal) intermediate
                              computational state needed to continue the process.
            metric_file_name: The file name of metric file, a csv file whose schema
                              specified by `metric_schema` plus "step".
            save_path: The path to which the new data files will be saved.
                       We will also search for data files in this path for
                       restoring purpose.
            restore_path: The path in which we will also search for data files for
                          restoring purpose. We will NOT write new data to this path.
            remote_storage_handler: A handler interacting with remote file system.
                                    By default, it's a dummy handler, meaning
                                    we will not interact with any remote system.
            local_storage_handler: A handler interacting with local file system.
            lazy_setup: Whether we do setup when initializing this object.
                        If False, we postpone the setup to the point when we enter
                        the context.
        '''

        self.block_size = block_size
        self.remote_storage_handler = remote_storage_handler
        self.local_storage_handler = local_storage_handler

        # The step corresponding to the "latest" safe and unsafe data.
        self.unsafe_step = None
        self.safe_step = None

        # Save path may be updated especially when no local save path is providied
        # but we need a local save path anyways. Same for restore path.
        self.save_path, self.restore_path = initialize_data_paths(save_path, restore_path,
                                                                  remote_storage_handler)
        self.metric_file_name = metric_file_name
        self.alive_metric_file_name = '__' + metric_file_name
        self.data_file_pattern = data_file_prefix + '_{step:06d}.tgz'
        self.data_index_pattern = data_file_prefix + f'_([0-9]+).tgz'
        self.ckpt_file_prefix = ckpt_file_prefix
        self.metric_schema = metric_schema

        # The initialization is postponed if `lazy_setup` is True.
        self.workspace_local_path = None
        self.ckpt_handler = None
        self.alive_metric_manager = None

        if lazy_setup:
            self.already_setup = False
        else:
            self.setup()
            self.already_setup = True

    def setup(self):
        '''
        1. Find the latest data to continue the process from. The found data
           is called "restore data".
        2. Initialize ckpt_handler and metric manager.
        '''
        self.workspace_local_path = self._make_workspace(self.save_path.local_path)

        self._download_latest_data(self.save_path.remote_path, self.save_path.local_path)
        self._download_latest_data(self.restore_path.remote_path, self.restore_path.local_path)

        self.restore_data_path, self.restore_step = self._find_latest_data(
            self.local_storage_handler,
            self.save_path.local_path, self.restore_path.local_path)

        self.ckpt_handler = CkptHandler(ckpt_file_prefix=self.ckpt_file_prefix,
                                        local_save_path=self.workspace_local_path)
        self.alive_metric_manager = MetricManager(file_name=self.alive_metric_file_name,
                                                  schema=self.metric_schema,
                                                  local_save_path=self.workspace_local_path)

    def __enter__(self):
        # download remote files
        # copy from restore to save path.
        # copy save path to workspace
        if not self.already_setup:
            self.setup()
        self.alive_metric_manager.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self.alive_metric_manager.__exit__(exception_type, exception_value, traceback)
        # This action is likely not necessary, but we will do it anyways erring
        # on the safe side.
        self._upload_safe_data(fail_if_not_found=False)

    @property
    def has_safe_data(self):
        '''
        Whether we can find any safe data.
        '''
        return self._has_data_at(self.safe_step)

    @property
    def has_unsafe_data(self):
        '''
        Whether we can find any unsafe data.
        Unsafe data will be promoted to safe after next block.
        '''
        return self._has_data_at(self.unsafe_step)

    @property
    def has_restore_data(self):
        '''
        Restore data

        '''
        return self._has_data_helper(self.restore_data_path)

    @property
    def local_safe_data_path(self):
        return self._get_data_path_in_workspace(self.safe_step)

    @property
    def local_unsafe_data_path(self):
        return self._get_data_path_in_workspace(self.unsafe_step)

    @property
    def local_alive_metric_path(self):
        return os.path.join(self.workspace_local_path, self.alive_metric_file_name)

    @property
    def local_metric_path(self):
        return os.path.join(self.workspace_local_path, self.metric_file_name)

    def run(self, step: int, ckpt_data: State, metric_data: List[Union[int, float]]):
        '''
        Write metric data.
        Save ckpt when approaching a new block. The newly saved data is an
        unsafe one, then the prevous unsafe data will be promoted as a safe one.
        Only the safe data will be save to the local and remote save path.
        '''

        self._write_metric(step=step, data=metric_data)

        # We will need to save unsafe ckpt even at the beginning.
        if step % self.block_size == 0:
            self._save_unsafe_and_commit_safe_data(step, ckpt_data)

    def restore_and_rollback(self) -> Tuple[int, State]:
        '''
        Restore previous safe data and roll back the state of the internal metric
        manager.
        Return the step and state from the loaded data.

        Raise `NoSafeDataAvailable` if no safe data is available.
        Raise `NotSetupYet` if this method is called before set up this CkptMetricManager.
        '''
        if not self.already_setup:
            raise NotSetupYet()

        self.alive_metric_manager.close()
        if not self.has_safe_data:
            raise NoSafeDataAvailable('No safe data to use for restore purpose!')

        # also need to update the file handle managed by the metric manager
        step, ckpt_data = self.load_latest_safe_data()
        self._rm_unsafe_data()
        self.unsafe_step = None
        return step, ckpt_data

    def load_restore_data(self) -> Tuple[Optional[int], Optional[State]]:
        '''
        Load the "restore data" found in save path and restore path.
        Return the step and state in the loaded data file.
        If no "restore data" found, then return a pair of None.
        If "restore data" is found, the internal metric manager will also load
        the corresponding metric file and append to it.

        Raise `NotSetupYet` if this method is called before set up this CkptMetricManager.
        '''
        if not self.already_setup:
            raise NotSetupYet()

        if not self.has_restore_data:
            return None, None

        # We may want to clean up the ckpt here.
        with tarfile.open(self.restore_data_path, "r:gz") as tar:
            tar.extractall(self.workspace_local_path)

        data = self.ckpt_handler.load_ckpt(self.restore_step)
        self.alive_metric_manager.reset(self.local_metric_path)
        return self.restore_step, data

    def load_latest_safe_data(self) -> Tuple[int, State]:
        '''
        Load the latest safe data produced in the QMC process.
        Return the step and state in the loaded data file.
        The internal metric manager will also load the corresponding metric file and append to it.

        Raise `NotSetupYet` if this method is called before set up this CkptMetricManager.
        '''
        if not self.already_setup:
            raise NotSetupYet()

        if not self.has_safe_data:
            raise NoSafeDataAvailable()

        with tarfile.open(self.local_safe_data_path, "r:gz") as tar:
            tar.extractall(self.workspace_local_path)

        data = self.ckpt_handler.load_ckpt(self.safe_step)
        self.alive_metric_manager.reset(self.local_metric_path)
        return self.safe_step, data

    # TODO Add unittests for this method and use it in the DMC process.
    def clean_up_save_path(self,
                           keep_latest_data_num: int = 3,
                           keep_every_N_steps: int = 10000,
                           worker_num: int = 1):
        '''
        Remove old data files from save path
        Args:
            keep_latest_data_num: The number of latest data fils to be kepts
            keep_every_N_steps: If a data file has step which is a multiplier
                                of `keep_every_N_steps`, then it will be kept.
            worker_num: How many working threads to be used in the clean-up procedure.
        '''
        self._clean_up(directory=self.save_path.remote_path,
                       storage_handler=self.remote_storage_handler,
                       keep_latest_data_num=keep_latest_data_num,
                       keep_every_N_steps=keep_every_N_steps,
                       worker_num=worker_num)
        self._clean_up(directory=self.save_path.local_path,
                       storage_handler=self.local_storage_handler,
                       keep_latest_data_num=keep_latest_data_num,
                       keep_every_N_steps=keep_every_N_steps,
                       worker_num=worker_num)

    def _save_unsafe_and_commit_safe_data(self, step: int, ckpt_data: State):
        '''
        Save ckpt when approaching a new block. The newly saved data is an
        unsafe one, then the prevous unsafe data will be promoted as a safe one.
        Only the safe data will be save to the local and remote save path.
        '''
        old_unsafe_step = self.unsafe_step
        old_safe_step = self.safe_step

        # Before we save new unsafe data, if we already have unsafe data,
        # then the that existing unsafe data will be promoted to safe data
        # afterwards and should be uploaded to save_path.
        should_upload_safe_data = self.has_unsafe_data

        self.alive_metric_manager.flush()
        unsafe_local_ckpt_path = self.ckpt_handler.save(step=step, state=ckpt_data)

        # Tarball current unsafe ckpt and alive metric file as the unsafe data.
        shutil.copy(self.local_alive_metric_path, self.local_metric_path)
        self.unsafe_step = step
        make_tarball(self.local_unsafe_data_path,
                     unsafe_local_ckpt_path,
                     self.local_metric_path)
        # After tarballing, the ckpt and metric file can be safely removed.
        self.ckpt_handler.rm_ckpt_at(step)
        os.remove(self.local_metric_path)


        # old_unsafe_step may be None due to retry mechanism in which case
        # the unsafe data are erased.
        # If previously we have unsafe data, then this unsafe data
        # should be commited and we should be able to upload it to save path.
        if should_upload_safe_data:
            self.safe_step = old_unsafe_step
            self._upload_safe_data(fail_if_not_found=True)

            if old_safe_step is not None:
                self._rm_data_in_workspace_at(old_safe_step)

    def _write_metric(self, step: int, data: List[Union[int, float]]):
        self.alive_metric_manager.write(step, data)

    def _make_workspace(self, save_path: str) -> str:
        '''
        Create a scrach space inside the local save path.
        '''
        workspace_local_path = os.path.join(save_path, '__workspace')
        if self.local_storage_handler.exists(workspace_local_path):
            self.local_storage_handler.rm(workspace_local_path)
        self.local_storage_handler.mkdir(workspace_local_path)
        return workspace_local_path

    def _has_data_helper(self, path: str) -> bool:
        '''
        Whether the given local `path` exists.
        '''
        return path is not None and self.local_storage_handler.exists(path)

    def _has_data_at(self, step: str) -> bool:
        '''
        Whether we have a data file indexed by the given `step`.
        '''
        return self._has_data_helper(self._get_data_path_in_workspace(step))

    def _upload_safe_data(self, fail_if_not_found: bool = True):
        '''
        Copy / Upload the latest safe data to local / remote save path.
        '''
        if not self.has_safe_data:
            msg = 'No Safe data found'
            if fail_if_not_found:
                raise Exception(msg)
            else:
                logging.warning(msg)
                return

        self.local_storage_handler.put(self.local_safe_data_path, self.save_path.local_path)
        if self.save_path.has_remote():
            self.remote_storage_handler.put(self.local_safe_data_path, self.save_path.remote_path)

    def _get_data_path_in_workspace(self, step: Optional[str]) -> Optional[str]:
        '''
        Get the path of a data file indexed by the given `step` in the "workspace".
        '''
        if step is None:
            return None
        return os.path.join(self.workspace_local_path, self.data_file_pattern.format(step=step))

    def _rm_data_in_workspace_at(self, step: str):
        '''
        Remove the path of a data file indexed by the given `step` in the "workspace".
        '''
        if self._has_data_at(step):
            data_path = self._get_data_path_in_workspace(step)
            self.local_storage_handler.rm(data_path)

    def _rm_unsafe_data(self):
        '''
        Remove the unsafe data in the "workspace".
        '''
        self._rm_data_in_workspace_at(self.unsafe_step)

    def _download_latest_data(self, remote_path: str, local_dir: str):
        '''
        Download the latest data file from the `remote_path` to the `local_dir`.
        '''
        if remote_path is None:
            return
        latest_ckpt_path, latest_index = self._find_latest_data(self.remote_storage_handler, remote_path)
        if latest_ckpt_path is not None:
            self.remote_storage_handler.get(latest_ckpt_path, local_dir)

    def _find_latest_data(self, storage_handler: StorageHandler, *directories: List[str]) -> Tuple[Optional[str], Optional[int]]:
        '''
        Find the latest data file from all the `directories`.
        Return the path of the latest data file and its step.

        If no data is found, return a pair of None
        '''

        paths = itertools.chain(*(storage_handler.ls(d, return_fullpath=True)
                                  for d in directories))

        latest_index = -1
        latest_path = None

        for path in paths:
            filename = os.path.basename(path)
            index = self._get_data_index(filename)
            # only download data
            if index is None:
                continue
            if index > latest_index:
                latest_index = index
                latest_path = path

        if not latest_path:
            return None, None
        return latest_path, latest_index

    def _get_data_index(self, filename):
        matched = re.match(self.data_index_pattern, filename)
        if matched is None:
            return None
        index_str, = matched.groups()
        return int(index_str)

    def _clean_up(self,
                  directory: str,
                  storage_handler: StorageHandler,
                  keep_latest_data_num: int,
                  keep_every_N_steps: int,
                  worker_num: int):
        if not directory:
            return

        all_paths = storage_handler.ls(directory, return_fullpath=True)
        to_remove = self._get_to_remove_paths(all_paths, keep_latest_data_num, keep_every_N_steps)

        if worker_num == 1:
            for f in to_remove:
                storage_handler.rm(f)
        else:
            with ThreadPoolExecutor(max_workers=worker_num) as executor:
                executor.map(storage_handler.rm, to_remove)

    def _get_to_remove_paths(self, all_paths, keep_latest_data_num, keep_every_N_steps):
        '''
	Get the paths to remove in the ckpt_clean_up process.
	We can keep K most recent ckpts (according to the ckpt index) controlled by `keep_latest_data_num`.
	Then we will remove the paths whose index is divisble by N, controlled by `keep_every_N_steps`.

	Args:
	  all_paths: A list of full paths for all the data to be considered.
	  keep_latest_data_num: Non-negative integer. How many most recent data that
			     will be kept.
	  keep_every_N_steps: Positive integer. Only remove the paths whose index
			      is not divisible by this argument
	Returns:
	  A list of full paths to be removed
	'''
        all_paths = []
        for full_path in all_paths:
            filename = os.path.basename(full_path)
            index = self._get_data_index(filename)
            if index is not None:
                all_paths.append((index, full_path))

        sorted_paths = sorted(all_paths, key=lambda x: x[0], reverse=True)
        result = []
        for index, path in sorted_paths[keep_latest_data_num:]:
            if index % keep_every_N_steps != 0:
                result.append(path)
        return result


def make_tarball(tarball_path: str, *paths: List[str]):
    '''
    Create a tar file at `tarball_path` consisting files specified by `paths`.
    '''
    with tarfile.open(tarball_path, "w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=os.path.basename(p))
