# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Handlers interacting with different types of storage systems.
'''

from abc import ABC, abstractmethod
import os
import pathlib
import re
import shutil
import subprocess
from typing import List

class StorageHandler(ABC):
    '''
    put, get, mv, cp would do overwrite when the destined path already exists.
    '''

    @abstractmethod
    def put(self, src_path: str, dst_path: str):
        """
        Put the file at local `src_path` to the `dst_path` at the file system
        handled by this handler.
        If `dst_path` already exists, then it will be overwritten.
        """
        pass

    @abstractmethod
    def get(self, src_path: str, dst_path: str):
        """
        Get the file from `src_path` at the file system handled by this handler
        to local `dst_path`.
        If `dst_path` already exists, then it will be overwritten.
        """
        pass

    @abstractmethod
    def rm(self, path: str):
        '''
        Should be able to remove both file and directory.
        '''
        pass

    @abstractmethod
    def ls(self, path: str, return_fullpath: bool = True) -> List[str]:
        '''
        If `return_fullpath` is True, then the return values should full path,
        otherwise it should only return file names.
        '''
        pass

    @abstractmethod
    def mkdir(self, path: str):
        pass

    @abstractmethod
    def mv(self, src_path: str, dst_path: str):
        """
        Both `src_path` and `dst_path` should be at the file system handled by this handler.
        If `dst_path` already exists, then it will be overwritten.
        """
        pass

    @abstractmethod
    def cp(self, src_path: str, dst_path: str):
        """
        Both `src_path` and `dst_path` should be at the file system handled by this handler.
        If `dst_path` already exists, then it will be overwritten.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        '''
        Whether `path`, file or directory, exists.
        '''
        pass

    @abstractmethod
    def exists_dir(self, directory: str) -> bool:
        '''
        Whether `directory` exists. Return True only when `directory` exists
        and it's actually a directory.
        '''
        pass

class DummyStorageHandler(StorageHandler):
    '''
    A Dummy implmentation of `RemoteStorageHandler` which does no-op for
    all the methods.
    '''

    @staticmethod
    def put(src_path: str, dst_path: str):
        return

    @staticmethod
    def get(src_path: str, dst_path: str):
        return

    @staticmethod
    def rm(path: str):
        return

    @staticmethod
    def ls(path: str, return_fullpath: bool = True) -> List[str]:
        return []

    @staticmethod
    def mkdir(path: str):
        return

    @staticmethod
    def mv(src_path: str, dst_path: str):
        return

    @staticmethod
    def cp(src_path: str, dst_path: str):
        return

    @staticmethod
    def exists(path: str) -> bool:
        return False

    @staticmethod
    def exists_dir(path: str) -> bool:
        return False

dummy_storage_handler = DummyStorageHandler()

class LocalStorageHandler(StorageHandler):

    @staticmethod
    def put(src_path: str, dst_path: str):
        LocalStorageHandler.cp(src_path, dst_path)

    @staticmethod
    def get(src_path: str, dst_path: str):
        LocalStorageHandler.cp(src_path, dst_path)

    @staticmethod
    def rm(path: str):
        if not os.path.exists(path):
            return
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    @staticmethod
    def ls(path: str, return_fullpath: bool = True) -> List[str]:
        if return_fullpath:
            all_paths = pathlib.Path(path).glob('*')
            return [str(f.absolute()) for f in all_paths]
        else:
            return os.listdir(path)

    @staticmethod
    def mkdir(path: str):
        os.makedirs(path)

    @staticmethod
    def mv(src_path: str, dst_path: str):
        shutil.move(src_path, dst_path)

    @staticmethod
    def cp(src_path: str, dst_path: str):
        shutil.copy(src_path, dst_path)

    @staticmethod
    def exists(path: str):
        return os.path.exists(path)

    @staticmethod
    def exists_dir(directory: str):
        exists = os.path.exists(directory)
        return exists and os.path.isdir(directory)

local_storage_handler = LocalStorageHandler()

class HdfsStorageHandler(StorageHandler):

    def __init__(self, command_prefix='', env_variables=None):
        self.command_prefix = command_prefix
        env_variables = {} if env_variables is None else env_variables
        self.env = dict(os.environ, **env_variables)

    def call_helper(self, *args, check_output=False):
        args = list(args)
        if not self.command_prefix:
            full_args = ['hdfs', 'dfs'] + args
        else:
            full_args = [self.command_prefix, 'hdfs', 'dfs'] + args

        if check_output:
            return subprocess.check_output(full_args, env=self.env)
        else:
            try:
                subprocess.check_call(full_args, env=self.env)
                return True
            except subprocess.CalledProcessError:
                return False

    def put(self, src_path: str, dst_path: str):
        self.call_helper('-put', '-f', src_path, dst_path)

    def get(self, src_path: str, dst_path: str):
        exists_file = (LocalStorageHandler.exists(dst_path)
                       and (not LocalStorageHandler.exists_dir(dst_path)))
        if exists_file:
            LocalStorageHandler.rm(dst_path)
        self.call_helper('-get', src_path, dst_path)

    def rm(self, path: str):
        self.call_helper('-rm', '-r', path)

    def ls(self, path: str, return_fullpath: bool = True) -> List[str]:
        # A sample line of the return value of `hdfs dfs -ls` is like
        # '-rw-r--r--   3 renweiluo supergroup  276979794 2021-12-29 00:39 hdfs:///user/renweiluo/test/qmcjax_ckpt_000000.npz'
        # So we basically extract the last part from it given it starts with 'hdfs:/'.
        #
        # However if `path` is not an absolute path, then the last part may not
        # start with 'hdfs:/'. That said, it should always contains the given `path`
        # and we use it as the pattern to extract the needed info.
        if not path:
            return []

        hdfs_path_pattern = f'.* ([^ ]*{path}[^ ]*)$'
        raw_output = self.call_helper('-ls', path, check_output=True).decode().split('\n')
        results = []
        for line in raw_output:
            if not line:
                continue
            matched = re.match(hdfs_path_pattern, line)
            if matched is None:
                continue
            full_path, = matched.groups()
            filename = os.path.basename(full_path)
            results.append(filename)

        if return_fullpath:
            return [os.path.join(path, f) for f in results]
        return results

    def mkdir(self, path: str):
        self.call_helper('-mkdir', '-p', path)

    def mv(self, src_path: str, dst_path: str):
        self.call_helper('-mv', src_path, dst_path)

    def cp(self, src_path: str, dst_path: str):
        self.call_helper('-cp', src_path, dst_path)

    def exists(self, path: str) -> bool:
        return self.call_helper('-test', '-e', path)

    def exists_dir(self, directory: str) -> bool:
        return self.call_helper('-test', '-d', directory)
