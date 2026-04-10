# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

from . import system

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
