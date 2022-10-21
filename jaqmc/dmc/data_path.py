# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
`DataPath` represents a pair of local and remote paths for data saving and restoring.
Also handle the path initialization logic.
'''

import os
import tempfile
from typing import Optional, Union, Tuple

import attr

from .storage_handler import StorageHandler

@attr.s(auto_attribs=True, frozen=True)
class DataPath:
    '''
    A wrapper for a local path and a remote path.
    '''
    local_path: Optional[str] = None
    remote_path: Optional[str] = None

    def has_local(self):
        return self.local_path is not None and self.local_path.strip()

    def has_remote(self):
        return self.remote_path is not None and self.remote_path.strip()

def initialize_data_paths(save_path: Optional[Union[DataPath, str]],
                          restore_path: Optional[Union[DataPath, str]],
                          remote_storage_handler: StorageHandler) -> Tuple[DataPath, DataPath]:
    '''
    Create local paths if not exist.

    Download data files from remote paths to the corresponding local ones if available.

    Return the updated save path and restore path.
    Those paths need to be updated when no local path is provided. We always
    need a local path especially for save path to keep newly produced data files.
    '''

    save_path = _resolve_path(save_path)
    restore_path = _resolve_path(restore_path)
    return _setup_path(save_path, restore_path, remote_storage_handler)

def _resolve_path(path: Optional[Union[DataPath, str]]) -> DataPath:
    if path is None:
        return DataPath()
    if isinstance(path, str):
        return DataPath(local_path=path)
    # isinstance(path, DataPath)
    return path

def _setup_path(save_path: DataPath,
                restore_path: DataPath,
                remote_storage_handler: StorageHandler) -> Tuple[DataPath, DataPath]:

    def setup(ckpt_path: DataPath):
        if ckpt_path.has_local():
            local_path = ckpt_path.local_path
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            elif not os.path.isdir(local_path):
                raise Exception(f'The path {local_path} already exists and it is not a directory')
        else:
            # If no local path is specified, we use a temporary directory.
            # This is necessary for both `save_path` (so that we can save
            # ckpts) and `restore_path` (so that we can download remote ckpts to
            # this path).
            local_path = tempfile.mkdtemp()

        if ckpt_path.has_remote():
            if not remote_storage_handler.exists_dir(ckpt_path.remote_path):
                if remote_storage_handler.exists(ckpt_path.remote_path):
                    raise Exception(f'The remote path {ckpt_path.remote_path} already exists and it is not a directory')
                else:
                    remote_storage_handler.mkdir(ckpt_path.remote_path)
        return DataPath(local_path, ckpt_path.remote_path)

    save_path = setup(save_path)
    restore_path = setup(restore_path)
    return save_path, restore_path
