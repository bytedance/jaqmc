# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
A show case for LapNet & penalty integrated loss.

LapNet: https://github.com/bytedance/LapNet
'''


import os
import sys
import time

from absl import app, logging
import jax
import jax.numpy as jnp
import kfac_jax
from kfac_jax import utils as kfac_utils
import numpy as np

from lapnet import base_config, checkpoint, curvature_tags_and_blocks, hamiltonian, mcmc, networks
from lapnet.utils import writers
from lapnet.train import init_electrons, make_should_save_ckpt

from jaqmc.loss.factory import build_func_state, make_loss
from jaqmc.loss import utils

def train(cfg):

  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons
  local_batch_size = cfg.batch_size
  signed_network, params, data, sharded_key = prepare(cfg, atoms, charges, nspins, local_batch_size)

  local_energy = hamiltonian.local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins)

  total_loss = make_loss(
      signed_network,
      local_energy,
      with_spin=cfg.optim.enforce_spin.with_spin,

      # Energy related
      clip_local_energy=cfg.optim.clip_el,
      rm_outlier=cfg.optim.rm_outlier,
      el_partition=cfg.optim.el_partition_num,
      local_energy_outlier_width=cfg.optim.local_energy_outlier_width,
      # Spin related
      nspins=nspins,
      spin_cfg=cfg.optim.enforce_spin,
  )
  func_state = build_func_state(step=kfac_utils.replicate_all_local_devices(0))

  val_and_grad = jax.value_and_grad(total_loss, argnums=0, has_aux=True)
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    fg = 1.0 * (t_ >= cfg.optim.lr.warmup)
    orig_lr = cfg.optim.lr.rate * jnp.power(
          (1.0 / (1.0 + fg * (t_ - cfg.optim.lr.warmup)/cfg.optim.lr.delay)), cfg.optim.lr.decay)
    linear_lr = cfg.optim.lr.rate * t_ / (cfg.optim.lr.warmup + (cfg.optim.lr.warmup == 0.0))
    return fg * orig_lr + (1 - fg) * linear_lr

  optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_state=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=utils.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
      )

  sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
  opt_state = optimizer.init(params, subkeys, data, func_state)

  time_of_last_ckpt = time.time()
  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  should_save_ckpt = make_should_save_ckpt(cfg)

  mcmc_step, mcmc_width, update_mcmc_width = prepare_mcmc(cfg, signed_network, local_batch_size)
  do_logging, writer_manager, write_to_csv = prepare_logging(cfg.optim.enforce_spin.with_spin, ckpt_save_path)

  with writer_manager as writer:
      for t in range(cfg.optim.iterations):

        sharded_key, mcmc_keys, loss_keys = kfac_jax.utils.p_split_num(sharded_key, 3)
        func_state.spin.step = kfac_utils.replicate_all_local_devices(t)
        data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

        # Optimization step
        params, opt_state, _, stats = optimizer.step(
            params=params,
            func_state=func_state,
            state=opt_state,
            rng=loss_keys,
            data_iterator=iter([data]),
            momentum=kfac_jax.utils.replicate_all_local_devices(jnp.zeros([])),
            damping=kfac_jax.utils.replicate_all_local_devices(jnp.asarray(cfg.optim.kfac.damping)))
        mcmc_width = update_mcmc_width(t, mcmc_width, pmove[0])

        do_logging(t, pmove, stats)
        write_to_csv(writer, t, pmove, stats)

	# Checkpointing
        if should_save_ckpt(t, time_of_last_ckpt):
            if cfg.optim.optimizer != 'none':
                checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width, sharded_key)
            time_of_last_ckpt = time.time()

def prepare(cfg, atoms, charges, nspins, local_batch_size):
  network_init, signed_network, *_ = networks.network_provider(cfg)(atoms, nspins, charges)
  key = jax.random.PRNGKey(int(1e6 * time.time()))

  params_initialization_key, sharded_key = jax.random.split(key)
  params = network_init(params_initialization_key)
  params = kfac_utils.replicate_all_local_devices(params)


  subkey, sharded_key = jax.random.split(sharded_key)
  data = init_electrons(subkey, cfg.system.molecule, cfg.system.electrons,
                        local_batch_size,
                        init_width=cfg.mcmc.init_width,
                        given_atomic_spin_configs=cfg.system.get('atom_spin_configs'))
  data = data.reshape([jax.local_device_count(), local_batch_size // jax.local_device_count(), *data.shape[1:]])
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(sharded_key)
  return signed_network, params, data, sharded_key

def prepare_mcmc(cfg, signed_network, local_batch_size):
  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(network, in_axes=(None, 0))
  mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      local_batch_size // jax.local_device_count(),
      steps=cfg.mcmc.steps,
      blocks=cfg.mcmc.blocks,
  )
  mcmc_step = utils.pmap(mcmc_step, donate_argnums=1)

  pmoves = np.zeros(cfg.mcmc.adapt_frequency)
  def update_mcmc_width(t, mcmc_width, pmove):
      if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.55:
          mcmc_width *= 1.1
        if np.mean(pmoves) < 0.5:
          mcmc_width /= 1.1
        pmoves[:] = 0
      pmoves[t%cfg.mcmc.adapt_frequency] = pmove
      return mcmc_width

  return mcmc_step, mcmc_width, update_mcmc_width

def prepare_logging(enforce_spin, save_dir):
    schema = ['energy', 'var', 'pmove']
    if enforce_spin:
        schema += ['spin', 'spin_var']
    message = '{t} ' + ' '.join(f'{key}: {{{key}:.4f}}' for key in schema)
    writer_manager = writers.Writer(
        name='result',
        schema=schema,
        directory=save_dir,
        iteration_key=None,
        log=False)

    def _prepare(t, pmove, stats):
        aux = stats['aux']
        vmc_aux = aux.vmc
        logging_dict = {
            't': t,
            'energy': stats['loss'][0],
            'var': vmc_aux.variance[0],
            'pmove': pmove[0]}

        if enforce_spin:
            spin_aux = aux.spin
            logging_dict['spin'] = spin_aux.estimator[0]
            logging_dict['spin_var'] = spin_aux.variance[0]
        return logging_dict

    def do_logging(t, pmove, stats):
        logging_dict = _prepare(t, pmove, stats)
        logging.info(message.format(**logging_dict))

    def write_to_csv(writer, t, pmove, stats):
        logging_dict = _prepare(t, pmove, stats)
        writer.write(**logging_dict)

    return do_logging, writer_manager, write_to_csv

def main(_):
    cfg = FLAGS.config
    cfg = base_config.resolve(cfg)
    loss_cfg = FLAGS.loss_config
    cfg['optim'] = {**cfg['optim'], **loss_cfg}

    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)
    train(cfg)

if __name__ == '__main__':
    from absl import flags
    from ml_collections.config_flags import config_flags
    import pathlib
    FLAGS = flags.FLAGS

    config_flags.DEFINE_config_file('config', None, 'Path to config file.')
    loss_config_file = str(pathlib.Path(os.path.abspath(__file__)).parents[1].absolute() / 'loss_config.py')
    config_flags.DEFINE_config_file('loss_config', loss_config_file, 'Path to loss config file.')
    app.run(main)
