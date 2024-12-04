# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import ml_collections

from jaqmc.loss import utils

DEFAULT_SPIN_PENALTY_CONFIG = ml_collections.ConfigDict({
    # Whether to enbale the spin-penalty term.
    # If True: Evaluate spin penalty in each step.
    'with_spin': False,
    # plus: Adopt S_+ penalty proposed in the paper arXiv:2406.01222 (2024).
    # square: Adopt S^2 penalty.
    'schema': 'plus',
    # Only works for S_+ penalty. Choose the component index of
    # wavefunction to anniliate for each step:
    # <\uparrow, ..., \downarrow, ...| S_{i,-} S_+ | \Psi> = 0.
    # If True: i is fixed as the first index of the electrons in the minor
    #   spin state.
    # If False: i is iteratively selected from the indices of electrons in
    #   the minor spin state along with the step.
    'fix_idx': False,
    # Partition walkers and calculate grad in loop to avoid out
    # of memory.
    # 1: calculate grad of all walkers at once.
    # other positive integer: Must to be divisible by the number of
    #   walkers. To partition walkers and calculate the grad
    #   iteratively, computing one partition at a time.
    'el_partition_num': 1,
    # Weight to add spin penalty in the loss.
    # Generally speaking, the value of weight should be proportional
    # to the spin-gap. Users should adjust the value according to
    # specific system working on, but trying the default value would be a
    # good starting point.
    # If weight <= 0, the penalty will not be added in the loss term.
    'weight': 2.,
    'rm_outlier': True,
    # Schema to remove outliers, could be 'deviation' or 'percentile'.
    'outlier_schema': 'deviation',
    'outlier_width': 10.,
    'outlier_bound': (1., 99.),
    'clip': {
        'schema': 'deviation',
        'width': 5.,
        'bound': (0., 100.),
    },
  }
)

@chex.dataclass
class SpinFuncState(utils.BaseFuncState):
    '''
    Attributes:
        step: The training step to determine which index of the electron to swap with.
          It will be updated by the penalty function and no need for extra care from users.
    '''
    step: int

@chex.dataclass
class SpinAuxData(utils.BaseAuxData):
  """
  Auxiliary data returned by spin penalty.

  Attributes:
    estimator: Estimation of spin-square value.
    swap_index: An index of the electron to swap with electrons in the opposite
      spin state.
    penalty_estimator: Estimator value of penalty, which is different with
      spin-square by a subtraction of Sz^2.
    penalty_local: Local value of permutation function,
      for `schema` of `spin_penalty` is `full`:
        \sum_alpha (\sum_beta -\hat{P}_{alpha, beta} + 1)
      for `schema` of `spin_penalty` is `approx`:
        N_alpha * (\sum_beta -\hat{P}_{alpha, beta} + 1)
      where alpha for spin states in minority and beta for spin state in
      majority.
    outlier_mask: Outlier mask for `penalty_local`
    variance: Mean variance of `penalty_local` over batch, and over all
      devices if inside a pmap.
  """
  estimator: jnp.ndarray
  swap_index: int
  penalty_estimator: jnp.ndarray
  penalty_local: jnp.ndarray
  outlier_mask: jnp.ndarray
  variance: jnp.ndarray

def make_spin_penalty(
  signed_network: utils.WaveFuncLike,
  local_energy: utils.LocalEnergy,
  nspins: Tuple,
  spin_cfg: ml_collections.ConfigDict=DEFAULT_SPIN_PENALTY_CONFIG,
  with_spin_grad=None,
) -> utils.Loss:
  """Creates the loss function with spin-penalty including the customized grad.

  Args:
    signed_network: A callable function representing the neural network wavefunction.
      The inputs include the parameters and coordinates.
      This function returns the log magnitude of the wavefunction together with its sign.
    local_energy: A callable function evaluating the local energy.
    nspins: A tuple containing numbers of spin-up and -down electrons.
    spin_cfg: ConfigDict containing parameters to deal with outliers and do
        clipping.
    with_spin_grad: whether we should include spin-related gradient in the gradient
        calculations. For instance, if we only want to check the spin value instead
        of enforcing spin symmetry, then we should set it to False. By default,
        it's "None", meaning we will use some internal logic to determine whether
        to include gradient or not.
  """

  # local_energy is not needed in the evaluation of spin penalty.
  del local_energy

  spin_schema = spin_cfg.schema
  el_partition_num = spin_cfg.el_partition_num
  spin_rm_outlier = spin_cfg.rm_outlier
  fix_idx = spin_cfg.fix_idx
  spin_outlier_schema = spin_cfg.outlier_schema
  spin_outlier_width = spin_cfg.outlier_width
  spin_outlier_bound = spin_cfg.outlier_bound
  spin_clip_schema = spin_cfg.clip.schema
  spin_clip_width = spin_cfg.clip.width
  spin_clip_bound = spin_cfg.clip.bound
  spin_weight = spin_cfg.weight

  if with_spin_grad is None:
      with_spin_grad = spin_cfg.weight > 0.

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
  spin_calculator = Spin(signed_network, nspins)
  minor_num = min(nspins)

  def make_spin_penalty_helper(schema, fix_idx):
  # schema 'square':
  #   `full_permute_wavefn` + sz * (sz + 1), where sz = abs(n_up - n_down) / 2.
  # 'approx' schema:
  #   `side_permute_wavefn` * N_\minor + sz * (sz + 1) ,
  #   where sz = abs(n_up - n_down) / 2 and N_\minor = min(n_up, n_down).

    def _square(params, data, idx):
      return spin_calculator.full_permute_wavefn(params, data)

    def _plus(params, data, idx):
      _val = spin_calculator.side_permute_wavefn(params, data, idx)
      return _val * minor_num

    def _plus_fix(params, data, idx):
      _idx = spin_calculator.idx_in_spin[spin_calculator.sz_polar - 1][0]
      _val = spin_calculator.side_permute_wavefn(params, data, _idx)
      return _val * minor_num

    if schema == 'square':
      return _square
    if schema == 'plus':
      if fix_idx:
        return _plus_fix
      else:
        return _plus

  spin_penalty = make_spin_penalty_helper(schema=spin_schema, fix_idx=fix_idx)

  spin_penalty_grad_vmapped = jax.vmap(
    jax.grad(spin_penalty, argnums=0), in_axes=(None, 0, None))

  def _spin_penalty_grad_body(params_and_idx, data):
    params, idx = params_and_idx
    return params_and_idx, spin_penalty_grad_vmapped(params, data, idx)

  fn_wavefn_grad_vmapped = jax.vmap(
    jax.grad(network, argnums=0), in_axes=(None, 0))

  def local_value_mask(
    local_value: jnp.ndarray,
    rm_outlier: bool = False,
    schema: str = 'deviation',
    outlier_width: float = 0.,
    outlier_bound: Tuple[float] = (1., 99.),
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Process the local value of an operator using a mask.

    Args:
      local_value: the local value to process.
      rm_outlier: If False, only `Nan` and `inf` in local values are masked.
         If True local_values considered as outlier will also be masked.
      schema:
        If 'deviation': The clipping edges are determined by the deviation from
          the mean of `local_value`.
        If 'percentile': The clipping edges are assigned as the specific
          percentile value of `local_value`.
      outlier_width: If assigned as 0, no local values except `Nan` and
        `inf` will be considered as outliers. If bigger than 0, local_value
        out of range [m - d * l, m + d * l] will be considered as outlier and
        masked, where l is this value, m is the mean of all local values, d
        is the mean of deviation of local value from m.
      outlier_bound: similar to `outlier_width` but effective for `percentile`.

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
            and (outlier_bound[0] != 0.)
            and (outlier_bound[1] != 100.)):
        # Gather local values from all hosts.
        val_all_hosts = utils.gather(local_value)

        # Local values out of the percentile range will be masked.
        lower_bd = jnp.percentile(val_all_hosts, outlier_bound[0])
        upper_bd = jnp.percentile(val_all_hosts, outlier_bound[1])
        mask = (
          (local_value > lower_bd) & (local_value < upper_bd) & is_finite)

    else:
      mask = is_finite

    return local_value, mask


  def local_value_clip(
    local_value: jnp.ndarray,
    schema: str = 'deviation',
    width: float = 5.,
    bound: Tuple[float] = (1., 99.),
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
      bound: A tuple containing the lower and upper percentile to clip.

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

      if bound[0] == 0. and bound[1] == 100.:
        return local_value

      # Gather local values from all hosts.
      val_all_hosts = utils.gather(local_value)

      # Local values out of the percentile range will be clipped.
      lower_bd = jnp.percentile(val_all_hosts, bound[0])
      upper_bd = jnp.percentile(val_all_hosts, bound[1])
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

  batch_spin_penalty = jax.vmap(spin_penalty, in_axes=(None, 0, None))
  batch_network = jax.vmap(network, in_axes=(None, 0))

  def fn_spin_penalty(
    params: utils.ParamTree,
    data: jnp.ndarray,
    step: int
  ) -> SpinAuxData:

    # Obtain the local value of spin penalty term, remove outliers and calculate
    # the estimation and variance of spin-square.
    _idx = step % minor_num
    idx = spin_calculator.idx_in_spin[spin_calculator.sz_polar - 1][_idx]
    spin_penalty_local = batch_spin_penalty(params, data, idx)

    spin_penalty_local, spin_penalty_mask = local_value_mask(
      spin_penalty_local,
      spin_rm_outlier,
      schema=spin_outlier_schema,
      outlier_width=spin_outlier_width,
      outlier_bound=spin_outlier_bound,
    )

    spin_penalty_estimator = utils.pmean_with_mask(spin_penalty_local, spin_penalty_mask)

    # sz * (sz + 1) is a fixed prefactor for spin square.
    sz = jnp.abs(nspins[0] - nspins[1]) * 0.5
    spin_estimator = sz * (sz + 1) + spin_penalty_estimator
    spin_var = utils.pmean_with_mask(
      (spin_penalty_local - spin_penalty_estimator) ** 2, spin_penalty_mask)

    return SpinAuxData(
      estimator=spin_estimator,
      swap_index=idx,
      penalty_estimator=spin_penalty_estimator,
      penalty_local=spin_penalty_local,
      outlier_mask=spin_penalty_mask,
      variance=spin_var,
    )

  @jax.custom_vjp
  def fn_loss_with_spin_penalty(
    params: utils.ParamTree,
    func_state: SpinFuncState,
    key: chex.PRNGKey,
    data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[SpinFuncState, SpinAuxData]]:

    spin_aux = fn_spin_penalty(params, data, func_state.step)

    # (WARNING) we DO NOT return loss value here and only focus on the gradient
    # to be used in optimization. Namely we don't really care about the actual
    # value of the spin penalty but only care about how it affects optimization
    # via gradient.
    return (
      0.0,
      (func_state, spin_aux),
    )

  def fn_spin_grad(params, data, aux_data, wf_logdet_grad):
    # to be consistent with grad shape, reshape according to partition.
    spin_penalty_local = aux_data.penalty_local
    _mask = aux_data.outlier_mask

    # Clip the local values of spin-penalty term.
    spin_penalty_local = local_value_clip(
      local_value=spin_penalty_local,
      schema=spin_clip_schema,
      width=spin_clip_width,
      bound=spin_clip_bound)

    # spin-peantly operator is not self-adjoint, the grad of the estimator
    # contains one more part, E[\nabla P_l]

    # Obtain the first part of the estimator grad.
    sp_estimator_grad_part1 = self_adjoint_estimator_grad(
      spin_penalty_local, wf_logdet_grad, _mask)

    # Obtain the second part of the estimator grad.
    # The grad of each local value with respect to params.
    _, sp_local_grad = jax.lax.scan(
      _spin_penalty_grad_body,
      (params, aux_data.swap_index),
      data.reshape((el_partition_num, -1, *data.shape[1:])))

    sp_local_grad = jax.tree_util.tree_map(
      lambda x: x.reshape([-1, *x.shape[2:]]), sp_local_grad)

    # mask inf and nan.
    mask_tree = jax.tree_util.tree_map(
      lambda x: jnp.isfinite(x), sp_local_grad)

    sp_local_grad = jax.tree_util.tree_map(
      lambda x: jnp.nan_to_num(x), sp_local_grad)

    mask_tree = jax.tree_util.tree_map(
      jax.tree_util.Partial(_product, mat=_mask), mask_tree)

    sp_estimator_grad_part2 = utils.pmean_with_structure_mask(
      sp_local_grad, mask_tree)

    total_grad = jax.tree_util.tree_map(
      lambda x, y: spin_weight * (2 * x + y) * aux_data.penalty_estimator,
        sp_estimator_grad_part1,
        sp_estimator_grad_part2)

    return total_grad

  def spin_penalty_bwd(res, g):

    aux_data, params, data, func_state = res
    wf_logdet_grad = fn_wavefn_grad_vmapped(params, data)

    spin_grad = fn_spin_grad(
      params, data, aux_data, wf_logdet_grad)

    grad_out = jax.tree_util.tree_map(lambda x: x * g[0], spin_grad)

    return grad_out, None, None, None

  def dummy_bwd(res, g):
    aux_data, params, data, func_state = res
    grad_out = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    return grad_out, None, None, None

  def fn_fwd(
    params: utils.ParamTree,
    func_state: SpinFuncState,
    key: chex.PRNGKey,
    data: jnp.ndarray,
  ):
    loss, (func_state, aux_data) = fn_loss_with_spin_penalty(
      params, func_state, key, data
    )

    # (WARNING) We are NOT doing KFAC registeration here and we rely on the
    # VMC part of the loss to do so.
    #
    # import kfac_jax
    # val = batch_network(params, data)
    # kfac_jax.register_normal_predictive_distribution(val[:, None])

    return (loss, (func_state, aux_data)), (aux_data, params, data, func_state)


  if with_spin_grad:
    fn_loss_with_spin_penalty.defvjp(fn_fwd, spin_penalty_bwd)
  else:
    fn_loss_with_spin_penalty.defvjp(fn_fwd, dummy_bwd)

  return fn_loss_with_spin_penalty

class Spin:

    def __init__(self, signed_network, nspins):
        """Initializes spin instance

        Args:
            signed_network: wavefunction with params and data as input and sign
                and logdet as the output
            nspins: A tuple of numbers of spin-up and -down electrons.
        """
        self.network = signed_network
        self.signed_network_vmapped = jax.vmap(signed_network, in_axes=(None, 0))
        self.nspins = nspins
        # Indices of all electrons.
        self.idx_list = jnp.arange(sum(nspins))
        # Electrons indices splitted by spin-up and -down.
        self.idx_in_spin = [jnp.arange(nspins[0]),
                            jnp.arange(nspins[0], sum(nspins))]

        #
        # Electrons spin polarization (the spin state in the
        # majority), 0 for up and 1 for down. If there are equal number
        # of spin-up and -down electrons, then take sz_polar = 1. So the
        # indices of spin-state in majority are stored in
        # `idx_list[sz_polar]` so that the indices of spin-state in
        # minority are stored in `idx_list[sz_polar - 1]`.
        if nspins[0] > nspins[1]:
            self.sz_polar = 0
        else:
            self.sz_polar = 1

    def side_permute_coord(self, data: jnp.ndarray, idx: int) -> jnp.ndarray:
        """Swap the coordinations of a single electron in the minority spin
        state with the coordinates of all electrons in the majority spin state.
        If there are equal numbers of spin-up and -down electrons, then swap
        one single spin-up electron with all spin-down electrons.

        Args:
            data: coordinations of one walker with n electrons (with the shape
                [n * 3]), or of a batch of m walkers (with the shape [m, n * 3])
            idx: index of the electron with spin state in minority,
                if nspins[0] > nspins[1]:
                    nspins[0] <= idx < nspins[1]
                else:
                    0 <= idx < nspins[0]

        Returns:
            permutations of coordinations with electron indiced idx swapped
            with all electrons in majority spin-state.
            if idx belongs to spin-up:
                the shape should be (nelectron[1], *data.shape)
            if idx belongs to spin-down:
                the shape should be (nelectron[0], *data.shape)
        """

        def swap(data: jnp.ndarray, idx1: int, idx2: int) -> jnp.ndarray:
            """Swap coordinations of two electrons for one walker or a batch
            of walkers.

            Args:
                data: coordinations of one walker with n electrons (the shape
                    should be n * 3) or of a batch with m walkers (the shape
                    should be [m, n * 3]).
                idx_1: one index of the electron pair to swap.
                idx_2: one index of the electron pair to swap.

            Return:
                coordinations of swapped walkers.
            """

            # reshape each walker by electrons
            data = data.reshape([*data.shape[:-1], -1, 3])

            # swap the index of idx1 and idx2 electron
            idx_list = self.idx_list.at[idx1].set(idx2)
            idx_list = idx_list.at[idx2].set(idx1)

            # flat coordinations of each walker
            return data[..., idx_list, :].reshape(*data.shape[:-2], -1)

        def _swap(data_and_idx1, idx2):
            return data_and_idx1, swap(*data_and_idx1, idx2)

        # swap the coordinations of idx electron with all electrons in
        # majority spin-state.
        _, data_permutation = jax.lax.scan(
            _swap, (data, idx), self.idx_in_spin[self.sz_polar])

        return jnp.stack(data_permutation)

    def side_permute_wavefn(self, params, data, idx) -> jnp.ndarray:
        """Sum over the wavefunction values of data in different permutations.
        For each permutation, the idx-th electron is swapped with one electron
        in majority spin-state.

        Args:
            params: parameters of the network.
            data: coordinations of a single walker.
            idx: electron index of the minority spin-state.

        Returns:
            \sum_{n_\alpha} -\hat{P}_{idx, n_\alpha} \Psi(r) + 1
            n_\alpha is the index of electrons in majority spin-state.
        """
        orig_sign, orig_logdet = self.network(params, data)

        # obtain the permutations of walkers with the idx-th electron swapped for the electrons
        # with opposite spin.
        data_permutation = self.side_permute_coord(data, idx)
        sign_permutation, logdet_permutation = self.signed_network_vmapped(
            params, data_permutation)
        sign = -sign_permutation * orig_sign
        logdet = logdet_permutation - orig_logdet
        val_max = jnp.max(logdet)
        logdet -= val_max
        val_permutation = jnp.sum(jnp.exp(logdet) * sign) * jnp.exp(val_max)
        return val_permutation + 1.

    def full_permute_wavefn(self, params, data) -> jnp.ndarray:
        """Sum over the wavefunction values of data in different permutations.
        For different permutation, one electron in mionrity spin-state is
        swapped with on electron in majority spin-state.

        Args:
            params: parameters of the network.
            data: coordinations of a single walker.

        Returns:
            \sum_{n_\beta} (\sum_{n_\alpha} - \hat{P}_{n_\alpha, n_\beta} \Psi(r) + 1)
            n_alpha > n_\beta
        """

        # first to swap one electron in minority spin-state with all electrons
        # in majority spin-state and then loop for all electrons in minority
        # spin-state
        side_batch_permute_wavefn = jax.vmap(
            self.side_permute_wavefn, in_axes=(None, None, 0))
        val_permutation = side_batch_permute_wavefn(
            params, data, self.idx_in_spin[self.sz_polar - 1])
        return jnp.sum(val_permutation)

    def spin_square(self, params, data) -> jnp.ndarray:
        """ calculate spin square,
        S^2 = sz * (sz + 1) + `self.full_permute_wavefn(params, data)`

        Return:
            local value of spin-square
        """
        sz = jnp.abs(self.nspins[0] - self.nspins[1]) * 0.5
        prefactor = sz * (sz + 1)
        val_permutation = self.full_permute_wavefn(params, data)
        return prefactor + val_permutation

    def spin_square_approx(self, params, data):
        """ approximation of spin square.
        S^2 \approx sz * (sz + 1) + N_\beta * `self.side_permute_wavefn(params, data, idx)`
        \alpha is assumed as the spin polarized direction and \beta is the opposite direction.
        idx \in \beta
        the estimation of spin-square should be the integration over coordinates rather than
        the local value.

        Returns:
            local value of the approximation.
        """
        sz = jnp.abs(self.nspins[0] - self.nspins[1]) * 0.5
        prefactor = sz * (sz + 1)

        # idx is choosen as the first index of the spin with less electrons
        idx = self.idx_in_spin[self.sz_polar - 1][0]

        val_permutation = self.side_permute_wavefn(params, data, idx)
        return prefactor + val_permutation * self.nspins[self.sz_polar - 1]
