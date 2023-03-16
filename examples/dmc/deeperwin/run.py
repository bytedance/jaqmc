# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
A show case for DeepErwin integration.
'''

import os
import pathlib
import sys
import time
import ruamel.yaml

from absl import app
from absl import logging
from deeperwin.utils import merge_params
from deeperwin.checkpoints import load_run
from deeperwin.configuration import Configuration
import jax
import jax.numpy as jnp

from jaqmc.dmc import run
from jaqmc.dmc.ckpt_metric_manager import DataPath

from deeperwin_model import build_log_psi_squared_with_sign

def run_wrapper(deeperwin_ckpt_file, dmc_cfg):

    reuse_data = load_run(deeperwin_ckpt_file)
    config = reuse_data.config

    # Build wavefunction and initialize parameters
    log_psi_squared_with_sign, _, params, fixed_params = \
        build_log_psi_squared_with_sign(config.model,
                                        config.physical,
                                        reuse_data.fixed_params)
    params = merge_params(params, reuse_data.params)

    nuclei = jnp.array(config.physical.R)
    charges = jnp.array(config.physical.Z)

    def log_psi_with_sign(x):
        sign, log_squared = log_psi_squared_with_sign(params,
                                                      x.reshape((-1, 3)),
                                                      nuclei,
                                                      charges,
                                                      fixed_params)
        return sign, log_squared / 2

    raw_position_shape = reuse_data.mcmc_state.r.shape
    position = reuse_data.mcmc_state.r.reshape(
        (raw_position_shape[0], raw_position_shape[1] * raw_position_shape[2]))

    key = jax.random.PRNGKey(int(1e6 * time.time()))
    run(
        position,
        dmc_cfg.iterations,
        log_psi_with_sign,
        dmc_cfg.time_step,
        key,
        nuclei=nuclei,
        charges=charges,

        # Below are optional arguments
        mixed_estimator_num_steps=dmc_cfg.mixed_estimator_num_steps,
        energy_window_size=dmc_cfg.energy_window_size,
        weight_branch_threshold=dmc_cfg.weight_branch_threshold,
        update_energy_offset_interval=dmc_cfg.update_energy_offset_interval,
        energy_offset_update_amplitude=dmc_cfg.energy_offset_update_amplitude,
        energy_cutoff_alpha=dmc_cfg.energy_cutoff_alpha,
        effective_time_step_update_period=dmc_cfg.effective_time_step_update_period,
        energy_outlier_rel_threshold=dmc_cfg.energy_outlier_rel_threshold,
        fix_size=dmc_cfg.fix_size,
        ebye_move=dmc_cfg.ebye_move,
        block_size=dmc_cfg.block_size,
        max_restore_nums=dmc_cfg.max_restore_nums,
        save_path=DataPath(dmc_cfg.log.save_path, dmc_cfg.log.remote_save_path),
    )

def main(_):
    deeperwin_ckpt = FLAGS.deeperwin_ckpt
    dmc_cfg = FLAGS.dmc_config

    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)

    run_wrapper(deeperwin_ckpt, dmc_cfg)

if __name__ == '__main__':
    from absl import flags
    from ml_collections.config_flags import config_flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('deeperwin_ckpt', '', 'NA')
    dmc_config_file = str(pathlib.Path(os.path.abspath(__file__)).parents[1].absolute() / 'dmc_config.py')
    config_flags.DEFINE_config_file('dmc_config', dmc_config_file, 'Path to DMC config file.')

    app.run(main)
