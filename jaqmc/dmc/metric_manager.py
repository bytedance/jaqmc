# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Handle metric csv file creation / writing / flushing / loading.
'''

import csv
import os
import shutil
from typing import List, Union

from absl import logging
import pandas as pd

class MetricManager:
    '''
    A context manager for the metric file.

    It support reset the underlying file using a given metric file, for instance,
    when we need to revert the state due to certain failure.
    '''

    def __init__(self,
                 file_name: str,
                 schema: List[str],
                 local_save_path: str):
        '''
        Args:
            file_name: The name of the metric file
            schema: The column names of the metric file. We will prepand "step" to it,
                    so no need to keep "step" in `schema`.
            local_save_path: The local directory storing the metric file
        '''
        self.file_name = file_name
        self.full_schema = ['step'] + schema
        self.local_metric_path = os.path.join(local_save_path, self.file_name)

        self.file_handle = None
        self.csv_writer = None

    def __enter__(self):
        self._setup_csv_writer()

    def __exit__(self, exception_type, exception_value, traceback):
        self.flush()
        if self.file_handle is not None:
            self.file_handle.close()

    def _setup_csv_writer(self):
        '''
        Create the file handle.

        Write schema to metric file if needed
        '''
        csv_should_write_header = not os.path.exists(self.local_metric_path)
        self.file_handle = open(self.local_metric_path, 'a+')
        self.csv_writer = csv.writer(self.file_handle, delimiter=',')
        if csv_should_write_header:
            self.csv_writer.writerow(self.full_schema)

    def flush(self):
        if self.file_handle is not None and not self.file_handle.closed:
            try:
                self.file_handle.flush()
            except Exception as e:
                logging.warning(f'Hitting error {e} when flushing metric file')

    def close(self):
        if self.file_handle is not None and not self.file_handle.closed:
            try:
                self.file_handle.close()
            except Exception as e:
                logging.warning(f'Hitting error {e} when closing metric file handle')

    def get_metric_data(self) -> pd.DataFrame:
        '''
        Return the content of the current metric file.
        '''
        self.flush()
        data = pd.read_csv(self.local_metric_path, names=self.full_schema, header=0)
        return data

    def write(self, step: int, data: List[Union[int, float]]):
        if self.csv_writer is not None:
            self.csv_writer.writerow([step] + list(data))

    def reset(self, target_path: str, rm_source: bool = True):
        '''
        Reset the managed metric file by the file at `target_path`.

        If `rm_source` is True, remove the file at `target_path` afterwards.
        '''
        self.close()
        if not self._samefile(target_path, self.local_metric_path):
            if rm_source:
                shutil.move(target_path, self.local_metric_path)
            else:
                shutil.copy(target_path, self.local_metric_path)
        self._setup_csv_writer()

    @staticmethod
    def _samefile(f1: str, f2: str) -> bool:
        if not os.path.exists(f1) or not os.path.exists(f2):
            return False
        return os.path.samefile(f1, f2)
