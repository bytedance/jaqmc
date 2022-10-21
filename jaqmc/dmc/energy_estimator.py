# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Energy mixed estimator calculation (stateful).
'''

from collections import deque

# The energy calculation is not relying on jit or pmap, so old numpy may be
# performing better.
import numpy as np

def append_wrapper(que, elem):
    if len(que) < que.maxlen:
        to_pop = None
    else:
        to_pop = que[0]
    que.append(float(elem))
    return to_pop

def update_deque_then_get_array(que, elem):
    que.append(float(elem))
    return np.array(que)

class MixedEstimatorCalculator:
    def __init__(self, mixed_estimator_num_steps, energy_window_size):
        self.mixed_estimator_num_steps = mixed_estimator_num_steps
        self.all_energy = deque([], energy_window_size)
        self.all_Pi_log = deque([], energy_window_size)
        self.all_total_weight = deque([], energy_window_size)
        self.all_energy_offsets = deque([], mixed_estimator_num_steps)

    def update_then_get_normalized_Pi(self, new_Pi_log):
        # Pi is likely to be exploded with large `mixed_estimator_num_steps`
        # or large `enery_window_size`. To resolve this issue, we remove the
        # largest `Pi_log` from all the `Pi_log` so that taking exponential of
        # those `Pi_log` won't make trouble. Note that such removal won't affect
        # the final result because `Pi` show up in both top and bottom of the mixed
        # estimator ratio, in which case removing a common factor doesn't change anything.
        self.all_Pi_log.append(float(new_Pi_log))
        all_Pi_log_array = np.array(self.all_Pi_log)
        max_Pi_log = np.max(all_Pi_log_array)
        all_Pi_log_array -= max_Pi_log
        return np.exp(all_Pi_log_array)

    @staticmethod
    def calculate_numerator(all_total_weight, all_Pi, all_energy):
        # Here `all_total_weight` is involved because we expect the energy
        # contained in `all_energy` are calculated with total weight divided,
        # namely the weight used to calculate energy is "normalized"
        return np.sum(all_energy * all_Pi * all_total_weight)

    @staticmethod
    def calculate_denominator(all_total_weight, all_Pi):
        return np.dot(all_total_weight, all_Pi)

    def get_Pi_log(self, time_step, energy_offset):
        self.all_energy_offsets.append(energy_offset)
        return -time_step * np.sum(self.all_energy_offsets)

    def run(self, energy_offset, energy, total_weight, time_step):
        '''
        Args:
            energy_offset: `E_T` used to do population control.
            energy: The weighted-averaged local energy. Here the weights used are
                    expected to be normalized (namely `w_i(t) / W(t)` where `W(t)`
                    is the total weight
            total_weight: Total weight `W(t)` as mentioned in Umrigar paper
        Return:
            The mixed estimator of energy.
        '''
        new_Pi_log = self.get_Pi_log(time_step, energy_offset)

        all_Pi = self.update_then_get_normalized_Pi(new_Pi_log)
        all_energy = update_deque_then_get_array(self.all_energy, energy)
        all_total_weight = update_deque_then_get_array(self.all_total_weight, total_weight)

        numerator = self.calculate_numerator(all_total_weight, all_Pi, all_energy)
        denominator = self.calculate_denominator(all_total_weight, all_Pi)
        return numerator / denominator
