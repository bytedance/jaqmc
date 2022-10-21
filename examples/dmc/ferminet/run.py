# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
A show case for FermiNet integration.
'''

import pathlib
import sys
import time

from absl import app
from absl import logging
from ferminet import base_config
from ferminet import checkpoint
from ferminet import envelopes
from ferminet import networks
import jax
import jax.numpy as jnp

from jaqmc.dmc import run
from jaqmc.dmc.ckpt_metric_manager import DataPath


def get_molecule_configuration(cfg):
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    spins = cfg.system.electrons
    return atoms, charges, spins

def run_wrapper(cfg, dmc_cfg):
    atoms, charges, spins = get_molecule_configuration(cfg)
    key = jax.random.PRNGKey(666 if cfg.debug.deterministic else int(1e6 * time.time()))

    envelope = envelopes.make_isotropic_envelope()
    _, network, *_ = networks.make_fermi_net(
        atoms, spins, charges,
        envelope=envelope,
        bias_orbitals=cfg.network.bias_orbitals,
        use_last_layer=cfg.network.use_last_layer,
        full_det=cfg.network.full_det,
        **cfg.network.detnet)

    vmc_ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
    vmc_ckpt_restore_filename = checkpoint.find_last_checkpoint(vmc_ckpt_save_path)
    _, data, params, *_ = checkpoint.restore(vmc_ckpt_restore_filename, cfg.batch_size)

    position = data.reshape((-1, data.shape[-1]))

    # Get a single copy of network params from the replicated one
    single_params = jax.tree_map(lambda x: x[0], params)
    network_wrapper = lambda x: network(params=single_params, pos=x)

    run(
        position,
        dmc_cfg.iterations,
        network_wrapper,
        dmc_cfg.time_step, key,
        nuclei=atoms,
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
    cfg = FLAGS.config
    cfg = base_config.resolve(cfg)
    dmc_cfg = FLAGS.dmc_config

    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)
    run_wrapper(cfg, dmc_cfg)

if __name__ == '__main__':
    from absl import flags
    from ml_collections.config_flags import config_flags
    FLAGS = flags.FLAGS

    config_flags.DEFINE_config_file('config', None, 'Path to config file.')

    dmc_config_file = str(pathlib.Path(__file__).parents[1].absolute() / 'dmc_config.py')
    config_flags.DEFINE_config_file('dmc_config', dmc_config_file, 'Path to DMC config file.')

    app.run(main)
