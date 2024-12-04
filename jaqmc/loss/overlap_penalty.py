from typing import List, Tuple

import chex
import jax
import jax.numpy as jnp
import ml_collections
from jaqmc.loss import utils

DEFAULT_OVERLAP_PENALTY_CONFIG = ml_collections.ConfigDict({
    # ConfigDict related with the overlap penalty.
    # Checkpoint path to restore the wavefunction being orthogonal
    # with. The checkpoint file should contain the mcmc walkers in
    # the keyword 'data', network parameters in the keyword 'params',
    # and the width of mcmc step in 'mcmc_width'.
    'restore_path': {
      # Local path, seperated by comma if multi paths are provided as
      # 'file1,file2'
      'local': '',
      # Path to save the downloaded checkpoint. If not provided,
      # checkpoints are saved in a timestamped directory in the
      # working directory.
      'save_path': '',
    },
    # A string to assign the network name. Different network names are
    # supposed to be seperated by comma in a consistent order with
    # `restore_path`, such as 'lapnet,lapnet'. If left blank, `lapnet`
    # is set for each network by default.
    'network_name': '',
    # A tuple containing `hidden_dims` of each network.
    # ((256, 4), (256, 4), (256, 4)) by default.
    'hidden_dims': (),
    # A tuple containing `weights` for all overlap penalty terms.
    # 0. by default.
    'weights': (),
    # MCMC steps for target states.
    'steps': 30,
    'clip_bond': (0., 100.),
  },
)

@chex.dataclass
class OverlapFuncState:
    overlap_data: jnp.ndarray

@chex.dataclass
class OverlapAuxData:
    overlap: Tuple[jnp.ndarray]
    mean_fixed_dist: List[float]
    div_current_dist: List[jnp.ndarray]
    mask: List[jnp.ndarray]

def make_overlap_penalty(
    signed_network: utils.LogWaveFuncLike,
    local_energy: utils.LocalEnergy,
    optim_cfg: ml_collections.ConfigDict,
    overlap_wf: List[utils.LogWaveFuncLike],
    overlap_weights: List[float],
):
  """ Overlap penalty term with customized grad.

  Args:
    signed_network: A callable function evaluating the probability distribution
      of the wavefunction. The inputs include the parameters and coordinates.
      This function returns the log value of the square root of the probability
      together with the sign.
    local_energy: A callable function evaluating the local energy.
    optim_cfg: ConfigDict containing parameters to deal with outliers and do
        clipping.
    overlap_wf:
        Collection of callable functions which are target wavefunctions to
        keep orthogonal with. The inputs should be the coordinates. The
        functions return the log probability together with the sign.
    overlap_weights: The collection of overlap penalty weights.

  """

  # local_energy is not needed in the evaluation of overlap penalty.
  del local_energy

  overlap_clip_bond = optim_cfg.orthogonal.clip_bond

  def _product(tree_leaf: jnp.ndarray, mat: jnp.ndarray) -> jnp.ndarray:
    """product a matrix with a tree_leaf of a pytree in a broadcasting style.

    Args:
      mat: with shape [n, m] where n is el_partition_num and m is
        num_walker // n
      tree_leaf: with shape [*mat.shape, ...], where ... represents the shape
        of the parameters in this leaf.

    Returns:
      with the same shape as tree_leaf.
    """
    mat = jnp.expand_dims(
      mat, axis=range(len(mat.shape), len(tree_leaf.shape)))

    return mat * tree_leaf

  network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  fn_wavefn_grad_vmapped = jax.vmap(
    jax.grad(network, argnums=0), in_axes=(None, 0))

  def local_value_mask(
    local_value: jnp.ndarray,
    rm_outlier: bool = False,
    schema: str = 'deviation',
    outlier_width: float = 0.,
    outlier_bond: Tuple[float] = (1., 99.),
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Process the local value of an operator using a mask.

    Args:
      local_value: the local value to process.
      rm_outlier: If False, only `Nan` and `inf` in local values are masked.
         If True local_values considered as outlier will also be masked.
      local_outlier_width: If assigned as 0, no local values except `Nan` and
        `inf` will be considered as outliers. If bigger than 0, local_value
        out of range [m - d * l, m + d * l] will be considered as outlier and
        masked, where l is this value, m is the mean of all local values, d
        is the mean of deviation of local value from m.

    Returns:
      local_value: local value with `Nan` and `inf` removed.
      mask: with the same shape as local value, 0 for masked local value and
        1 for not masked local value.
    """

    is_finite = jnp.isfinite(local_value)
    local_value = jnp.nan_to_num(local_value)

    if rm_outlier:
      if schema == 'deviation' and outlier_width > 0.:
        val_mean = utils.pmean(jnp.mean(local_value))
        dev_mean = utils.pmean(jnp.mean(jnp.abs(local_value - val_mean)))

        mask = (
          (val_mean - outlier_width * dev_mean < local_value) &
          (val_mean + outlier_width * dev_mean > local_value) &
          is_finite)

      elif (schema == 'percentile'
            and (outlier_bond[0] != 0.)
            and (outlier_bond[1] != 100.)):
        # Gather local values from all hosts.
        # The data communication may be potentially expensive.
        val_all_hosts = utils.gather(local_value)

        # Local values out of the percentile range will be masked.
        lower_bd = jnp.percentile(val_all_hosts, outlier_bond[0])
        upper_bd = jnp.percentile(val_all_hosts, outlier_bond[1])
        mask = (
          (local_value > lower_bd) & (local_value < upper_bd) & is_finite)

    else:
      mask = is_finite

    return local_value, mask


  def local_value_clip(
    local_value: jnp.ndarray,
    schema: str = 'deviation',
    width: float = 5.,
    bond: Tuple[float] = (1., 99.),
  ) -> jnp.ndarray:
    """ Clip local values of an operator with respect to Monte Carlo walkers.

    Args:
      local_value: The local value to process.
      schema:
        If 'deviation': The clipping edges are determined by the deviation from
          the mean of `local_value`.
        If 'percentile': The clipping edges are assigned as the specific
          percentile value of `local_value`.
      width: The width to do clipping based on the deviation.
        If <=0: The local value will not be clipped and return `local_value`
          directly.
        If >0:
          The clipping edges are set as: [m - n * d, m + n * d], where m is the
          mean of local values, n is this value and D is the mean of deviation
          of local value from m.
      bond: A tuple containing the lower and upper percentile to clip.

    Returns:
      local_value: Has been clipped `local_value`.
    """

    if schema == 'deviation':

      if width <= 0.:
        return local_value

      val_mean = utils.pmean(jnp.mean(local_value))
      dev_mean = utils.pmean(jnp.mean(jnp.abs(local_value - val_mean)))

      local_value = jnp.clip(local_value,
                             val_mean - width * dev_mean,
                             val_mean + width * dev_mean)

    elif schema == 'percentile':

      if bond[0] == 0. and bond[1] == 100.:
        return local_value

      # Gather local values from all hosts.
      # The data communication may be potentially expensive.
      val_all_hosts = utils.gather(local_value)

      # Local values out of the percentile range will be clipped.
      lower_bd = jnp.percentile(val_all_hosts, bond[0])
      upper_bd = jnp.percentile(val_all_hosts, bond[1])
      local_value = jnp.clip(local_value, lower_bd, upper_bd)

    return local_value


  def self_adjoint_estimator_grad(
      local_value: jnp.ndarray,
      wf_logdet_grad: utils.ParamTree,
      mask: jnp.ndarray,
  ) -> utils.ParamTree:
    """ Calculate grad of an estimator E[P_l] of a self-adjoint operator P.
    \nabla E[P] = E[(P_l - E[P_l]) * \nabla log(Psi)], where P_l is the local
    value of operator P

    Args:
      local_value: Local value of the self-adjoint operator, which is P_l
      wf_logdet_grad: Grad of wavefunction logdet value, which is
        \nabla log(Psi)
      mask: Mask of local values.

    Returns:
      estimator_grad: Grad of the estimator with respect to params.
    """

    val_mean = utils.pmean_with_mask(local_value, mask)
    diff = local_value - val_mean
    estimator_grad = jax.tree_util.tree_map(
      jax.tree_util.Partial(_product, mat=diff), wf_logdet_grad)

    mask_tree = jax.tree_util.tree_map(lambda x: jnp.isfinite(x), estimator_grad)
    estimator_grad = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), estimator_grad)

    # the final mask should be mask * mask_tree
    # if one local energy is `nan` or `inf`, we should mask all the parameters'
    # grad of the walker, if only the grad of one parameter is `nan` or `inf`,
    # we only mask the corresponding parameter.
    mask_tree = jax.tree_util.tree_map(
      jax.tree_util.Partial(_product, mat=mask), mask_tree)
    estimator_grad = utils.pmean_with_structure_mask(estimator_grad, mask_tree)

    return estimator_grad


  batch_signed_network = jax.vmap(signed_network, in_axes=(None, 0))

  batch_overlap_wf = []
  for _overlap_wf in overlap_wf:
    batch_overlap_wf.append(jax.vmap(_overlap_wf, in_axes=(0)))

  def fn_overlap_penalty(
    params: utils.ParamTree,
    data: jnp.ndarray,
    data_fixed_dist: List[jnp.ndarray],
  ) -> OverlapAuxData:
    """ Calculate the overlap of the current state and given target states.
          \hat{O} = E_{Psi_t}[Psi(R) / Psi_t(R)] * E_{Psi}[Psi_t(R) / Psi(R)]
        Where Psi_t is the target state to be orthogonal with, E_{Psi} means
        expectation value with walkers sampled according to the distribution of Psi(R)^2.

    Args:
      params: Parameter of the current state.
      data: Walkers sampled from the distribution of current state.

    Returns:
      OverlapAuxData: Chex class contains `overlap` and `div`.
        overlap: List of overlaps between current state and target states.
        div: [(E_{Psi_t}[Psi(R) / Psi_t(R)],
               E_{Psi}[Psi_t(R) / Psi(R)],
               Psi_t(R) / Psi(R)),
               ...]
          Each tuple cooresponds to a target state.
    """

    def div_in_dist(batch_wf1, batch_wf2, data):
      sign_1, logdet_1 = batch_wf1(data)
      sign_2, logdet_2 = batch_wf2(data)
      sign, log_diff = sign_1 * sign_2, logdet_1 - logdet_2

      # make `inf` and `nan` in `log_diff` to be 0.
      log_diff, mask = local_value_mask(log_diff)
      log_diff = log_diff * mask

      log_diff = local_value_clip(
        local_value=log_diff,
        schema='percentile',
        bond=overlap_clip_bond,
      )

      # Subtract the max value of `log_diff` to reduce numerical error.
      max_val = jnp.max(utils.gather(jnp.max(log_diff)))
      log_diff_res = log_diff - max_val
      prefactor = jnp.exp(max_val)
      signed_div_res = sign * jnp.exp(log_diff_res)

      # Obtain the mean value with `nan` and `inf` maksed.
      div_mean = utils.pmean_with_mask(signed_div_res, mask) * prefactor
      div_local = signed_div_res * prefactor

      return div_mean, div_local, mask

    batch_signed_wf = lambda data: batch_signed_network(params, data)
    overlap, mean_fixed_dist, div_current_dist, mask = [], [], [], []

    for _data_fixed_dist, _batch_overlap_wf in zip(data_fixed_dist,
                                                   batch_overlap_wf):

      _mean_fixed_dist, _, _ = div_in_dist(
        batch_signed_wf, _batch_overlap_wf, _data_fixed_dist)

      _mean_current_dist, _div_current_dist, _mask = div_in_dist(
        _batch_overlap_wf, batch_signed_wf, data)

      overlap.append(_mean_fixed_dist * _mean_current_dist)
      mean_fixed_dist.append(_mean_fixed_dist)
      div_current_dist.append(_div_current_dist)
      mask.append(_mask)

    return OverlapAuxData(overlap=overlap,
                          mean_fixed_dist=mean_fixed_dist,
                          div_current_dist=div_current_dist,
                          mask=mask)

  @jax.custom_vjp
  def fn_loss_with_overlap_penalty(
    params: utils.ParamTree,
    func_state: OverlapFuncState,
    key: chex.PRNGKey,
    data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[OverlapFuncState, OverlapAuxData]]:

    overlap_aux = fn_overlap_penalty(params, data, func_state.overlap_data)

    # (WARNING) we DO NOT return loss value here and only focus on the gradient
    # to be used in optimization. Namely we don't really care about the actual
    # value of the overlap penalty but only care about how it affects optimization
    # via gradient.
    #
    # (TODO) Have a flag to control this behavior if we also become interested
    # in the actual value of overlap penalty.
    return (
      0.0,
      (func_state, overlap_aux),
    )

  def fn_overlap_grad(aux_data, wf_logdet_grad):
    means = aux_data.mean_fixed_dist
    divs = aux_data.div_current_dist
    masks = aux_data.mask

    overlap_grad = self_adjoint_estimator_grad(divs[0], wf_logdet_grad, masks[0])
    total_grad = jax.tree_util.tree_map(
      lambda x: overlap_weights[0] * means[0] * x, overlap_grad)

    for _mean, _div, _mask, _weight in zip(
      means[1:], divs[1:], masks[1:], overlap_weights[1:]
    ):
      overlap_grad = self_adjoint_estimator_grad(_div, wf_logdet_grad, _mask)
      total_grad = jax.tree_util.tree_map(
        lambda x, y: x + _weight * _mean * y, total_grad, overlap_grad)

    return total_grad


  def overlap_penalty_bwd(res, g):

    aux_data, params, data, func_state = res
    wf_logdet_grad = fn_wavefn_grad_vmapped(params, data)

    overlap_grad = fn_overlap_grad(aux_data, wf_logdet_grad)
    grad_out = jax.tree_util.tree_map(lambda x: x * g[0], overlap_grad)

    return grad_out, None, None, None

  def fn_fwd(
    params: utils.ParamTree,
    func_state: OverlapFuncState,
    key: chex.PRNGKey,
    data: jnp.ndarray,
  ):
    loss, (func_state, aux_data) = fn_loss_with_overlap_penalty(
      params, func_state, key, data
    )

    # (WARNING) We are NOT doing KFAC registeration here and we rely on the
    # VMC part of the loss to do so.
    #
    # import kfac_jax
    # val = batch_network(params, data)
    # kfac_jax.register_normal_predictive_distribution(val[:, None])

    return (loss, (func_state, aux_data)), (aux_data, params, data, func_state)

  fn_loss_with_overlap_penalty.defvjp(fn_fwd, overlap_penalty_bwd)
  return fn_loss_with_overlap_penalty
