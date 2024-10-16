# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Iterable, Mapping, Tuple, Union
from typing_extensions import Protocol

import chex
import jax
import jax.numpy as jnp
import kfac_jax

PMAP_AXIS_NAME = 'qmc_pmap_axis'

def wrap(func):
  return functools.partial(func, axis_name=PMAP_AXIS_NAME)

pmap = wrap(jax.pmap)
pmean = wrap(kfac_jax.utils.pmean_if_pmap)
psum = wrap(kfac_jax.utils.psum_if_pmap)
gather = wrap(jax.lax.all_gather)

def pmean_with_mask(value, mask):
  '''
  Only take pmean with the not-masked-out value (namely mask > 0). Here `mask`
  is expected to only take value between 0 and 1.
  '''
  return psum(jnp.sum(value * mask)) / psum(jnp.sum(mask))

def pmean_with_structure_mask(value, mask):
  '''
  Only take pmean with the not-masked-out value (namely mask > 0). Here `mask`
  is expected to only take value between 0 and 1.
  '''
  def inner(x, y):
    return psum(jnp.sum(x * y, axis=(0))) / psum(jnp.sum(y, axis=(0)))

  value_masked_mean = jax.tree_util.tree_map(inner, value, mask)
  return value_masked_mean

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]

class LocalEnergy(Protocol):

  def __call__(self, params: ParamTree, key: chex.PRNGKey,
               data: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: walkers consisting of electronic configurations.
    """

class WaveFuncLike(Protocol):

  def __call__(self, params: ParamTree,
               electrons: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions.
    """

class LogWaveFuncLike(Protocol):

  def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions
    """

@chex.dataclass
class BaseFuncState:
    """
    Base class for func_state passed to the loss function.
    """
    pass

@chex.dataclass
class BaseAuxData:
    """
    Base class for auxillary data returned from the loss function.
    """
    pass

class Loss(Protocol):

  def __call__(self,
               params: ParamTree,
               func_state: BaseFuncState,
               key: chex.PRNGKey,
               data: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[BaseFuncState, BaseAuxData]]:
    """
    Note: kfac_jax.optimizer.Optimizer should turn on flags `value_func_has_rng=True` and
    `value_func_has_aux=True` when working with loss functions of this interface.

    Args:
      params: network parameters.
      func_state: function state passed to the loss function to control its behavior.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    Returns:
      (loss value, (updated func_state, auxillary data)
    """
