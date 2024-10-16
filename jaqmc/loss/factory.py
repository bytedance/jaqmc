# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Optional

from absl import logging
import chex
import jax
import jax.numpy as jnp

from jaqmc.loss import utils
from jaqmc.loss.spin_penalty import SpinAuxData, SpinFuncState, make_spin_penalty
from jaqmc.loss.vmc import VMCAuxData, VMCFuncState, make_vmc_loss

@chex.dataclass
class FuncState:
    '''
    Parent Func State containing ones for each loss component.
    '''
    vmc: Optional[VMCFuncState] = None
    spin: Optional[SpinFuncState] = None

@chex.dataclass
class AuxData:
  '''
  Parent auxiliary data containing ones for each loss component.
  '''
  vmc: Optional[VMCAuxData] = None
  spin: Optional[SpinAuxData] = None

def build_func_state(step=None) -> FuncState:
    '''
    Helper function to create parent FuncState from actual data.
    '''
    if step is None:
        spin = None
    else:
        spin = SpinFuncState(step=step)

    return FuncState(
        vmc=None,
        spin=spin,
    )

def make_loss(
    signed_network: utils.WaveFuncLike,
    local_energy: utils.LocalEnergy,

    # Flags to control loss behavior by selecting loss components.
    with_spin=False,

    # kwargs for each loss components.
    **kwargs
):
    '''
    User-facing loss factory.

    Note: kfac_jax.optimizer.Optimizer should turn on flags `value_func_has_rng=True` and
    `value_func_has_aux=True` when working with the output loss function.

    Args:
      signed_network: Callable taking in params and data, returning sign
          and log-abs of the neural network wavefunction.
      local_energy: callable which evaluates the local energy.

      with_spin: If True, then add spin penalty in loss.

      kwargs: Other flags to be passed to each loss component.

    Returns:
        Callable with signature (params, func_state, key, data) and returns (loss, (func_state, aux_data))
    '''
    def invoke(component_factory):
        relevant_kwargs = _get_relevant_kwargs(kwargs, component_factory)
        logging.info(f'arguments for {component_factory.__name__} are {relevant_kwargs}')
        component_func = component_factory(
            signed_network,
            local_energy,
            **relevant_kwargs)
        return component_func

    # A list of pairs (loss_func, loss_identifier)
    all_components = []

    loss_func = invoke(make_vmc_loss)
    all_components.append([loss_func, 'vmc'])

    if with_spin:
        spin_penalty_func = invoke(make_spin_penalty)
        all_components.append([spin_penalty_func, 'spin'])

    def total_loss(
            params: utils.ParamTree,
            func_state: FuncState,
            key: chex.PRNGKey,
            data: jnp.ndarray
        ):
        _loss = 0.0
        all_func_state = dict()
        all_aux = dict()
        for func, component_key in all_components:
            key, sub_key = jax.random.split(key)
            primal, (new_func_state, aux) = func(params, func_state[component_key], sub_key, data)
            _loss += primal
            all_func_state[component_key] = new_func_state
            all_aux[component_key] = aux
        return _loss, (FuncState(**all_func_state), AuxData(**all_aux))
    return total_loss

def _get_relevant_kwargs(all_kwargs, func):
    '''
    Pick keyword arguments from `all_kwargs` that is relevant to `func`. Namely,
    only pick the ones that's belong to the `func`'s argument list.
    '''
    def __get_args(func):
        sig = inspect.signature(func)
        return set(p.name for p in sig.parameters.values())

    func_args = __get_args(func)
    return {k: v for (k, v) in all_kwargs.items() if k in func_args}
