# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Ckpt saving / loading
'''

import os
from typing import Optional, Tuple

from absl import logging
import numpy as np

from .state import State

class CkptHandler:
    '''
    Handling saving and loading checkpoint (ckpt for short) files.
    Only support local paths.
    '''
    def __init__(self,
                 ckpt_file_prefix: str,
                 local_save_path: str):
        '''
        Args:
            ckpt_file_prefix: The name of is ckpt file consists of two parts:
                              1. this `ckpt_file_prefix`
                              2. the step corresponding to this ckpt.
            local_save_path: Where to save and load ckpts.
        '''
        self.ckpt_file_prefix = ckpt_file_prefix
        self.ckpt_file_pattern = self.ckpt_file_prefix + '_{step:06d}.npz'
        self.local_save_path = local_save_path

    def load_ckpt(self, step: int) -> State:
        '''
        Load the ckpt file corresponding to `step`
        '''
        local_path = self._get_ckpt_path(step)
        _, state = self._load_ckpt(local_path)
        return state

    #TODO Have unittest to enforce the step in the name of ckpt file is
    #     the same as the one stored inside the ckpt.
    def save(self, step: int, state: State) -> str:
        '''
        Save the `state` in a ckpt corresponding to `step`.

        Return the local path of the saved ckpt file.
        '''
        local_path = self._get_ckpt_path(step)
        with open(local_path, 'wb') as f:
            np.savez(
                f,
                step=step,
                state=state)
        return local_path

    def rm_ckpt_at(self, target_step: Optional[int]):
        if self.local_save_path is None or target_step is None:
            logging.warning(f'No ckpt to remove at step {target_step} in path {self.local_save_path}')
            return
        to_rm_filename = self.ckpt_file_pattern.format(step=target_step)
        to_rm_path = os.path.join(self.local_save_path, to_rm_filename)
        if os.path.exists(to_rm_path):
            os.remove(to_rm_path)

    def _get_ckpt_path(self, step: int) -> str:
        return os.path.join(self.local_save_path, self.ckpt_file_pattern.format(step=step))

    @staticmethod
    def _load_ckpt(ckpt_path: str) -> Tuple[int, State]:
        '''
        Return the step and state in the ckpt file at `ckp_path`.
        '''
        ckpt_data = np.load(ckpt_path, allow_pickle=True)
        step = ckpt_data['step'].tolist()
        state = ckpt_data['state'].tolist()
        return step, state
