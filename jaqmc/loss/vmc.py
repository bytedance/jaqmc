# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import kfac_jax

from jaqmc.loss import utils

@chex.dataclass
class VMCFuncState:
    pass

@chex.dataclass
class VMCAuxData(utils.BaseAuxData):
  """
  Auxiliary data returned from energy calculation.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
    outlier_mask: boolean array representing whether certain walker is marked
                  as outlier in VMC calculation.
  """
  variance: jnp.ndarray
  local_energy: jnp.ndarray
  outlier_mask: jnp.ndarray

def make_vmc_loss(
  signed_network: utils.WaveFuncLike,
  local_energy: utils.LocalEnergy,
  clip_local_energy=0.0,
  rm_outlier=False,
  el_partition=1,
  local_energy_outlier_width=0.0) -> utils.Loss:
  """
  Creates the loss function corresponding to the energy calculation.

  Args:
    signed_network: network wavefunction returning both sign and log magnitude.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    rm_outlier: If True, outliers will be removed from the computation from both
      loss and its gradients, otherwise outliers would be clipped when
      computing gradients, in which case clipping won't happen in the computation
      of the loss value.
    el_partition: Create N folds data when computing local_energy to save the memory.
    local_energy_outlier_width: If greater than zero, the local energy outliers
      will be identified as the ones that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. Those outliers will be removed from the calculation
      of both the energy and its gradient, if `rm_outlier` is True.
  Returns:
    Callable with signature (params, data) and returns (loss, (None, aux_data)),
    where loss is the mean energy, and aux_data is of VMCAuxData.
    The loss is averaged over the batch and over all devices inside a pmap.
  """
  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, 0), out_axes=0)
  batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)


  @jax.custom_jvp
  def total_energy(
      params: utils.ParamTree,
      func_state: None,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[None, VMCAuxData]]:
    """
    Evaluates the total energy of the neural network wavefunction..

    Args:
      params: parameters of the neural network wavefunction.
      func_state: To pass the variables to be updated in training loop.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, (None, aux_data)), where loss is the averaged energy, and `None` for
      func_state, we don't need func_state so we just return `None`, aux_data
      is an AuxiliaryLossData object containing the variance of the energy and
      the local energy per MCMC configuration. The loss and variance are
      averaged over the batch and over all devices inside a pmap.
    """

    # we don't have any variable to be updated in training loop.
    del func_state

    keys = jax.random.split(key, num=data.shape[0])
    if el_partition > 1 :
      # create el_partition folds to save the memory when computing local energy
      data = data.reshape((el_partition,-1)+data.shape[1:])
      keys = keys.reshape((el_partition,-1)+keys.shape[1:])
      def batch_el_scan(carry,x):
        return carry, batch_local_energy(params, *x)
      _,e_l = jax.lax.scan(batch_el_scan, None, [keys,data])
      e_l = e_l.reshape(-1)
    else:
      e_l = batch_local_energy(params, keys, data)
    # is_finite is false for inf and nan. We should throw them away anyways.
    is_finite = jnp.isfinite(e_l)
    # Then we convert nan to 0 and inf to large numbers, otherwise we won't
    # be able to mask them out. It's ok to do this cast because they will be
    # masked away in the following computation.
    e_l = jnp.nan_to_num(e_l)

    # if not `rm_outlier`, which means we will do clipping instead, in which case
    # we don't clip when computing the energy but do clip in gradient computation.
    if rm_outlier and local_energy_outlier_width > 0.:
      # This loss is computed only for outlier computation
      loss = utils.pmean_with_mask(e_l, is_finite)
      tv = utils.pmean_with_mask(jnp.abs(e_l - loss), is_finite)
      mask = (
        (loss - local_energy_outlier_width * tv < e_l) &
        (loss + local_energy_outlier_width * tv > e_l) &
        is_finite)
    else:
      mask = is_finite

    loss = utils.pmean_with_mask(e_l, mask)
    variance = utils.pmean_with_mask((e_l - loss)**2, mask)

    return loss, (None, VMCAuxData(variance=variance,
                                   local_energy=e_l,
                                   outlier_mask=mask))

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):
    """Custom Jacobian-vector product for unbiased local energy gradients."""

    # func_state is not needed and assigned as `_`.
    params, _, key, data = primals
    loss, (func_state, aux_data) = total_energy(params, None, key, data)

    if clip_local_energy > 0.0:
      # We have to gather the el from all devices and then compute the median
      # otherwise the median would be different on different devices
      median = jnp.median(utils.gather(aux_data.local_energy))

      # We have to apply mask here to remove the effect of possible inf and nan.
      tv = utils.pmean_with_mask(jnp.abs(aux_data.local_energy - median), aux_data.outlier_mask)
      diff = jnp.clip(aux_data.local_energy,
                      median - clip_local_energy * tv,
                      median + clip_local_energy * tv)
      # renormalize diff
      diff = diff - utils.pmean_with_mask(diff, aux_data.outlier_mask)
      device_batch_size = jnp.sum(aux_data.outlier_mask)
    else:
      diff = aux_data.local_energy - loss
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
    diff *= aux_data.outlier_mask

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    primals = primals[0], primals[3]
    tangents = tangents[0], tangents[3]
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
    primals_out = loss, (func_state, aux_data)

    tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, (func_state, aux_data))
    return primals_out, tangents_out

  return total_energy
