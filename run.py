# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
A show case for LapNet with PH and ECP.

LapNet: https://github.com/bytedance/LapNet
'''


import sys
import time
from typing import Sequence, Tuple

from absl import app, logging
import jax
import jax.numpy as jnp
import kfac_jax
from kfac_jax import utils as kfac_utils
import ml_collections
import numpy as np
import pyscf

from lapnet import base_config, checkpoint, curvature_tags_and_blocks, hamiltonian, mcmc, networks 
from lapnet.utils import writers, system
from lapnet.train import make_should_save_ckpt

from jaqmc.loss.factory import build_func_state, make_loss
from jaqmc.loss import utils

def train(cfg):

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(
        pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons
  local_batch_size = cfg.batch_size
  
  signed_network, params, data, sharded_key, network_options = prepare(cfg, atoms, charges, nspins, local_batch_size)

  if use_ecp_or_ph(cfg):
      from jaqmc.pp.hamiltonian import pp_energy 
      logging.info('Applying ECP or PH from JaQMC')
      local_energy = pp_energy(
        f=signed_network,
        atoms=atoms,
        nspins=nspins,
        charges=charges,
        pyscf_mol=cfg.system.pyscf_mol,
        pp_cfg=cfg.ecp,
        energy_local=None,
        use_scan=False,
        el_partition_num=cfg.optim.el_partition_num,
        forward_laplacian=cfg.optim.forward_laplacian,
        )
  else:
      local_energy = hamiltonian.local_energy(
        f=signed_network,
        atoms=atoms,
        charges=charges,
        nspins=nspins)

  total_loss = make_loss(
      signed_network,
      local_energy,

      # Energy related
      clip_local_energy=cfg.optim.clip_el,
      rm_outlier=cfg.optim.rm_outlier,
      el_partition=cfg.optim.el_partition_num,
      local_energy_outlier_width=cfg.optim.local_energy_outlier_width,
      nspins=nspins,
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
  do_logging, writer_manager, write_to_csv = prepare_logging(ckpt_save_path)

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

def pyscf_mol_to_internal_representation(
    mol: pyscf.gto.Mole) -> ml_collections.ConfigDict:
  """Converts a PySCF Mole object to an internal representation.

  Args:
    mol: Mole object describing the system of interest.

  Returns:
    A ConfigDict with the fields required to describe the system set.
  """
  # Ensure Mole is built so all attributes are appropriately set.
  mol.build()
  atoms = [
      system.Atom(mol.atom_symbol(i), mol.atom_coord(i), charge=mol.atom_charge(i))
      for i in range(mol.natm)
  ]
  return ml_collections.ConfigDict({
      'system': {
          'molecule': atoms,
          'electrons': mol.nelec,
      },
      'pretrain': {
          # If mol.basis isn't a string, assume that mol is passed into
          # pretraining as well and pretraining uses the basis already set in
          # mol, rather than complicating the configuration here.
          'basis': mol.basis if isinstance(mol.basis, str) else None,
      },
  })


def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width=1.0,
    given_atomic_spin_configs: Sequence[Tuple[int, int]] = None
) -> jnp.ndarray:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3).
  """
  if given_atomic_spin_configs is None:
    logging.warning('no spin assignment in the system config, may lead to unexpected initialization')

  if (sum(atom.charge for atom in molecule) != sum(electrons)
      and
      given_atomic_spin_configs is None):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:

    atomic_spin_configs = [
            (atom.element.nalpha - int((atom.atomic_number - atom.charge) // 2),
             atom.element.nbeta - int((atom.atomic_number - atom.charge) // 2))
            for atom in molecule
    ] if given_atomic_spin_configs is None else given_atomic_spin_configs

    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      if atomic_spin_configs[i][0] > 0:
          atomic_spin_configs[i] = nalpha - 1, nbeta + 1

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  return (
      electron_positions +
      init_width *
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


def prepare(cfg, atoms, charges, nspins, local_batch_size):
  network_init, signed_network, network_options, *_ = networks.network_provider(cfg)(atoms, nspins, charges)
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
  return signed_network, params, data, sharded_key, network_options

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

def use_ecp_or_ph(cfg):
    if cfg.system.get('pyscf_mol') is None:
        return False
    # Including PH
    use_ecp = bool(cfg.system.pyscf_mol._ecp)
    if 'ecp' not in cfg or cfg.ecp.ph_info is None:
        use_ph = False
    else:
        use_ph = len(cfg.ecp.ph_info[0]) > 0
    logging.info(f'Use_ECP (including PH): {use_ecp}; Use_PH: {use_ph}')
    return use_ecp or use_ph

def prepare_logging(save_dir):
    schema = ['energy', 'var', 'pmove']
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

    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)
    train(cfg)

if __name__ == '__main__':
    from absl import flags
    from ml_collections.config_flags import config_flags
    import pathlib
    FLAGS = flags.FLAGS

    config_flags.DEFINE_config_file('config', None, 'Path to config file.')
    app.run(main)