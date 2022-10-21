# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
DMC configurations.
'''

import ml_collections

def get_config():
    # Some terminology / notation may refer to
    # Umrigar C J, Nightingale M P, Runge K J. A diffusion Monte Carlo algorithm with very small time‐step errors[J]. The Journal of chemical physics, 1993, 99(4): 2865-2890.

  cfg = ml_collections.ConfigDict({
          # Number of DMC iterations to run.
          # Note that in practice we will run more iterations (by one more "block")
          # then the one specified here just to make sure that
          # the data in the final checkpoint is outlier-free.
          'iterations': 10000,
          # DMC time step. Should be
          # 1. small enough to avoid finite-time error
          # 2. not too small otherwise the DMC process will last way too long.
          'time_step': 0.001,
          # The energy offset E_T will be updated every `update_energy_offset_interval`
          # iterations.
          'update_energy_offset_interval': 1,
          # The amplitude of adjustment on E_T given the weight calculated in each step.
          'energy_offset_update_amplitude':1,
          # Metric info will be printed out every `print_info_interval`.
          'print_info_interval': 20,
          # The lower and upper limits for branching / merging.
          'weight_branch_threshold': (0.3, 2),
          # The T_p used in the calculation of \Pi to adjust the effect of updating E_T.
          'energy_window_size': 1000,
          # The size of rolling window in which the energy mixed estimator is calculated.
          'mixed_estimator_num_steps': 5000,
          # The relative threshold to determine whether some calculated local
          # energies are outlier, in which case we will rerun the iteration with
          # a different random number wishing for better luck.
          # Negative value means turning off such mechanism.
          'energy_outlier_rel_threshold': -1.0,
          # If `energy_cutoff_alpha` < 0, then fallback to UNR algo. for
          # weight update, otherwise do energy cutoff following
          # Zen A, Sorella S, Gillan M J, et al. Boosting the accuracy and speed of quantum Monte Carlo: Size consistency and time step[J]. Physical Review B, 2016, 93(24): 241118.
          'energy_cutoff_alpha': 0.2,
          # Whether do fix-size branching or not.
          # It can be turned on to boost efficiency due to better JAX jitting.
          'fix_size': False,
          # If True, use elec-by-elec moves rather than walker-by-walker moves.
          # By default it's turned off due to efficiency concern.
          'ebye_move': False,
          # Negative `effective_time_step_update_period` means always
          # update effective time step.
          'effective_time_step_update_period': -1,

          # The size of a block of iterations. The recovery mechanism will
          # roll back to the previous block when error happens.
          'block_size': 5000,
          # The max number of rolling-back which the recovery mechasim will
          # perform before it gives up and abort the process.
          'max_restore_nums': 3,

          'log': {
              # The local path that the checkpoint will be saved to.
              'save_path': '',
              # The remote path that the checkpoint will be upload to.
              'remote_save_path': '',
              # The local path that the previous checkpoint will be loaded from.
              'restore_path': '',
              # The remote path that the previous checkpoint will be downloaded from.
              'remote_restore_path': '',
          },
      }
  )
  return cfg
