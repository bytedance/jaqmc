# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Entry point of the DMC implementation.
'''

import time
from typing import Optional, Union

from absl import logging
import attr
import jax
import jax.numpy as jnp
import numpy as np

from .branch import branch as branch_vary_size
from .branch_fix_size import branch as branch_fix_size
from .ckpt_metric_manager import CkptMetricManager
from .data_path import DataPath
from .effective_time_step_calculator import EffectiveTimeStepCalculator
from .energy_estimator import MixedEstimatorCalculator
from .hamiltonian import local_energy as build_local_energy_func, make_calc_energy_func
from .position_update import from_orig_config_wrapper, from_updated_config_wrapper, from_orig_config_wrapper_ebye
from .state import State, IterationOutput
from .storage_handler import dummy_storage_handler
from .utils import agg_mean, agg_sum

def run(init_position,
        num_steps,
        vmc_wave_func_with_sign,
        time_step,
        key,
        nuclei,
        charges,
        t_init=0,
        local_energy_func=None,
        velocity_func=None,
        mixed_estimator_num_steps=5000,
        energy_window_size=1000,
        weight_branch_threshold=(0.3, 2),
        anchor_energy=None,
        update_energy_offset_interval=1,
        energy_offset_update_amplitude=1,
        energy_cutoff_alpha=0.2,
        effective_time_step_update_period=-1,
        energy_clip_pair=None,
        energy_outlier_rel_threshold=-1,
        fix_size=False,
        ebye_move=False,
        block_size=5000,
        max_restore_nums=3,

        # Below are  Debug/Logging related arguments:
        debug_mode=False,
        state: State=None,
        save_path: Optional[Union[DataPath, str]] = None,
        restore_path: Optional[Union[DataPath, str]] = None,
        print_info_interval=10,
        output_csv_filename='metric.csv',
        ckpt_prefix='dmc_ckpt',
        data_prefix='dmc_data',
        remote_storage_handler=dummy_storage_handler):
    """
    The entry point of DMC

    Args:
        init_position: Initially sampled walkers' positions, usually sampled from
                       some related VMC.
                       Expected shape: (num_walkers, num_electrons * 3).
        num_steps: How many steps/iterations will be run in the DMC
        vmc_wave_func_with_sign: A function takes in a single configuration of electrons
                                 (a jnp array of shape (num_electrons * 3, )) and outputs a pair of scalar:

                                 (sign of wavefunction value,
                                  logarithm of absolute value of wavefunction)

                                 Note that this function is only for a single
                                 walker instead of a batch of walkers.
                                 DMC will take care of the batch-processing (via
                                 vmap and pmap) by itself.
        time_step: DMC time step.
        key: RNG key.
        nuclei, charges: The position and charges of all the nuclei. Both are
                         expected to be jnp arrays of shape (num_atoms, ).
        t_init: The initial step index.
        local_energy_func: An optional function taking in a configuration of electrons
                           (same as `vmc_wave_func_with_sign`) and outputing the corresponding local energy.
                           Similar to `vmc_wave_func_with_sign`, it should only
                           deal with a single walker.
                           If not given, then use the default implementation
                           of local energy calculation on `vmc_wave_func_with_sign`.
        velocity_func: An optional function to calculate the drift-term-related velocity
                       at each electron configuration. If not given, then use
                       the default implementation with JAX autograd.
        mixed_estimator_num_steps: How many renomarlization factor should be reverted
                                   when calculating the state at each time.
                                   Namely the `T_p` in `Pi(t_hat, T_p)` when calculating
                                   the mixed estimator in Umrigar et al 1993.
                                   Usually should be greater than the autocorrelation
                                   time of the averaged local energy.
        energy_window_size: How many terms will be averaged when calculating
                            the mixed estimator.
        weight_branch_threshold: A pair of lower and upper limits
                                 governing walkers branching / merging.
        anchor_energy: An anchor value used to determine outliers in calculated
                       local energy. If not provided, then use the energy from the
                       trial wavefunction.
        update_energy_offset_interval: How frequently shall we update the energy offset
        energy_offset_update_amplitude: How strongly do we expect the energy offset
                                        to constrain the variance of number of walkers.
        energy_clip_pair: (clip_min, clip_max) to be used in weight update and energy
                          calculation to clip the local energy. If None then no clipping
                          will be done.
                          Do note that if the guiding wave function is not well-trained,
                          in which case the fluctuation of the local energy could
                          be quite large and naive clipping may significantly affect
                          the DMC calculation.
                          If not provided, then no clipping will be done at all.
        energy_outlier_rel_threshold:
            If negative, not effective, otherwise we construct a "acceptance interval" around
            the current best energy estimator with width equal to the (absoluate) value of
            the energy estimator times this config value. If in the current run, the averaged
            energy is outside such "acceptance interval", then we reject the current iteration
            and rerun it with a different random number.
        fix_size: Whether do fix-size branching or not.
                  It can be turned on to boost efficiency due to better JAX jitting.
        ebye_move: If True, use elec-by-elec moves rather than walker-by-walker moves.
                   By default it's turned off due to efficiency concern.
        block_size: The size of a block of iterations. The recovery mechanism will
                    roll back to the previous block when error happens.
        max_restore_nums: The max number of rolling-back which the recovery mechasim will
                          perform before it gives up and abort the process.

    Debug/Logging related arguments:
        debug_mode: If true: will also store the averaged energy / weight / debug_info
                    in every step. It may occupy large GPU memory and trigger OOM eventually.
                    So only turn on for debugging purpose.
        state: The state with which the DMC will continue. Mainly for debug purpose.
        save_path: Can be of type str or `DataPath`.
                   string: Local path that the checkpoint will be saved to.
                   `DataPath`: Containing both local save_path and remote save_path.
        restore_path: Can be of type str or `DataPath`.
                      string: The local path that the previous checkpoint will be loaded from.
                      `DataPath`: Containing both local restore_path and remote restore_path.
        print_info_interval: Every how many steps do we print some information to stdout.
        data_prefix: The prefix of data tarball containing both checkpoint and the metric csv file.
        output_csv_filename: The filename of the metric csv file (will be included in the data tarball).
        ckpt_prefix: The prefix of checkpoint file (will be included in the data tarball).
        remote_storage_handler: The storage handler to interact with remote storage system.
    """
    vmc_wave_func = lambda x: vmc_wave_func_with_sign(x)[1]
    calc_energy_func = make_calc_energy_func(vmc_wave_func, nuclei, charges, clip_pair=energy_clip_pair)


    if energy_window_size < 0:
        energy_window_size = num_steps

    curr_num_walkers, _ = init_position.shape

    if local_energy_func is None:
        local_energy_func = build_local_energy_func(vmc_wave_func, nuclei, charges)

    if velocity_func is None:
        velocity_func = jax.grad(vmc_wave_func)

    metric_schema = ["estimator",
                     "offset",
                     "average",
                     "num_walkers",
                     "old_walkers",
                     "total_weight",
                     "acceptance_ratio",
                     "effective_time_step",
                     "num_cutoff_updated",
                     "num_cutoff_orig"]
    ckpt_metric_manager = CkptMetricManager(
        metric_schema=metric_schema,
        block_size=block_size,
        data_file_prefix=data_prefix,
        ckpt_file_prefix=ckpt_prefix,
        metric_file_name=output_csv_filename,
        save_path=save_path,
        restore_path=restore_path,
        remote_storage_handler=remote_storage_handler,
        lazy_setup=False)

    if state is None:

        # If `state` is None, we first check if there's any ckpt available.
        # If so, we load both ckpt data and metric file.
        t_init_from_ckpt, state_from_ckpt = ckpt_metric_manager.load_restore_data()
        if state_from_ckpt is not None:
            t_init = t_init_from_ckpt + 1
            state = state_from_ckpt
            logging.info(f'Continue DMC process with step {t_init}')
        else:
            # Both `state` and `state_from_ckpt` are None, fall back to some default value.
            state = State.default(
                init_position=init_position,
                calc_energy_func=calc_energy_func,
                mixed_estimator_num_steps=mixed_estimator_num_steps,
                energy_window_size=energy_window_size,
                time_step=time_step)
            logging.info(f'Starting DMC from default state')

    (flatten_position, flatten_walker_age, flatten_weight, flatten_local_energy,
     energy_offset, target_num_walkers, mixed_estimator,
     mixed_estimator_calculator, effective_time_step_calculator) = attr.astuple(state, recurse=False)

    if anchor_energy is None:
        anchor_energy = energy_offset

    # If we do fix_size DMC, we don't need to worry about the fluctuation of
    # the number of walkers, in which case we don't need to do padding, assuming
    # in the begining we can guarantee that the number of walkers can be
    # divided by the number of devices.
    do_padding = not fix_size
    step = make_step(
        vmc_wave_func_with_sign, velocity_func, local_energy_func,
        time_step,
        nuclei=nuclei,
        charges=charges,
        energy_cutoff_alpha=energy_cutoff_alpha,
        energy_clip_pair=energy_clip_pair,
        do_padding=do_padding,
        ebye_move=ebye_move)

    if mixed_estimator_calculator is None:
        mixed_estimator_calculator = MixedEstimatorCalculator(
            mixed_estimator_num_steps=mixed_estimator_num_steps,
            energy_window_size=energy_window_size)
    if effective_time_step_calculator is None:
        effective_time_step_calculator = EffectiveTimeStepCalculator(time_step)

    if fix_size:
        logging.info('Will do branching with fixed population size')
        branch = branch_fix_size
    else:
        logging.info('Will do branching with variable population size')
        branch = branch_vary_size

    dmc_single_iteration = make_dmc_single_iteration(
        time_step=time_step,
        dmc_step=step,
        branch=branch,
        weight_branch_threshold=weight_branch_threshold,
        energy_clip_pair=energy_clip_pair,
        update_energy_offset_interval=update_energy_offset_interval,
        energy_offset_update_amplitude=energy_offset_update_amplitude,
        energy_outlier_rel_threshold=energy_outlier_rel_threshold,
        ebye_move=ebye_move)

    key, subkey = jax.random.split(key)
    if max_restore_nums > 0:
        dmc_single_iteration = recovery_wrapper(
            dmc_single_iteration,
            max_restore_nums=max_restore_nums,
            ckpt_metric_manager=ckpt_metric_manager,
            key=subkey)

    all_energy = []
    all_weight = []
    all_position = []
    all_debug_info = []

    t = t_init
    with ckpt_metric_manager:
        while t < num_steps + block_size + 1:
            curr_time = time.time()
            # If `effective_time_step_update_period` is negative, we always update
            # effective time step. Otherwise we only update it until the specified
            # positive `effective_time_step_update_period`.
            should_update_effective_time_step = (
                (effective_time_step_update_period <= 0)
                or (t < effective_time_step_update_period))
            # This is the main step running DMC. All others are the ones
            # 1. prepare data / parameter for this step
            # 2. processing output from this function.
            # `output` is of type IterationOutput
            new_t, output = dmc_single_iteration(
                 index=t,
                 key=key,
                 state=state)
            (succeeded, state, key, averaged_energy,
             num_old_walkers, acceptance_ratio, effective_time_step,
             debug_info) = attr.astuple(output, recurse=False)

            (flatten_position, flatten_walker_age, flatten_weight, flatten_local_energy,
             energy_offset, target_num_walkers, mixed_estimator,
             mixed_estimator_calculator, effective_time_step_calculator) = attr.astuple(state, recurse=False)

            if not succeeded:
                logging.warning(f'Failed try for {t}-th step')
                t = new_t
                continue

            curr_num_walkers = int(agg_sum(flatten_position.shape[0]))
            # Remove the padded elements
            num_cutoff_updated = agg_sum(debug_info[-1].reshape(-1)[:flatten_position.shape[0]])
            num_cutoff_orig = agg_sum(debug_info[-2].reshape(-1)[:flatten_position.shape[0]])

            if debug_mode:
                all_energy.append(averaged_energy)
                all_debug_info.append(debug_info)
                all_weight.append(flatten_weight)
                all_position.append(flatten_position)

            total_weight = agg_sum(flatten_weight)


            ckpt_metric_manager.run(step=t,
                                    ckpt_data=state,
                                    metric_data=[mixed_estimator, energy_offset, averaged_energy,
                                       curr_num_walkers, num_old_walkers, total_weight,
                                       acceptance_ratio, effective_time_step,
                                       num_cutoff_updated, num_cutoff_orig])

            if t % print_info_interval == 0:
                total_weight = agg_sum(flatten_weight)
                logging.info(f'{t}, {time.time() - curr_time}, '
                    + f'Current energy estimator: {mixed_estimator}. '
                    + f'Updating energy offset to {energy_offset}. '
                    + f'Current number of walkers: {curr_num_walkers}, '
                    + f'total weight: {total_weight}, '
                    + f'acceptance ratio: {acceptance_ratio}, '
                    + f'effective_time_step: {effective_time_step}, '
                    + f'cutoff: {num_cutoff_updated}, {num_cutoff_orig}')
            t = new_t

    return (flatten_position, flatten_walker_age,
            all_energy, all_weight, all_position,
            all_debug_info)

def make_dmc_single_iteration(time_step,
                              dmc_step,
                              branch,
                              weight_branch_threshold,
                              energy_clip_pair,
                              update_energy_offset_interval,
                              energy_offset_update_amplitude,
                              energy_outlier_rel_threshold=-1,
                              ebye_move=False):
    '''
    Mainly handle
    1. Update walkers' positions and weights
    2. Do branching if necessary
    3. calculate energy

    Args:
        dmc_step: A function to update walkers' positions and weights
        mixed_estimator_calculator: handling energy calculation
        weight_branch_threshold: Thresholds governing when to split/merge walkers.
        energy_clip_pair: clipping limits to prevent outlier local energy from
                          ruining the average
        Check out the docstring of ```run``` function for other args.
    Return:
        A function doing each iteration of DMC.
    '''

    def is_energy_outlier(energy, anchor):
        '''
        if the newly calculated energy deviates from anchor energy too much,
        then we believe its an outlier
        '''
        if energy_outlier_rel_threshold <= 0.0:
            return False

        boundaries = [anchor * (1 - energy_outlier_rel_threshold),
                      anchor * (1 + energy_outlier_rel_threshold)]
        return (energy < min(boundaries)) or (energy > max(boundaries))

    def dmc_single_iteration(
            index,
            key,
            state: State,
            should_update_effective_time_step=True) -> IterationOutput:
        '''
        Args:
            index: The index in the DMC iteration, namely t_hat in Umrigar paper.
            flatten_position: Walkers' positions. Shape: (Batch, 3)
            flatten_walker_age: Walkers' age, to track if any persistent configuration.
                                Shape: (Batch,)
            flatten_weight: Walker's weight. Shape: (Batch,)
            key: RNG key
            prev_energy_offset: The energy offset calculated from the previous
                                step and to be used in the current step.
                                This is for population control.
            prev_mixed_estimator: Similar to `prev_energy_offset` but for mixed_estimator.
                                  This is the best best estimator of the energy so far.
        Return:
            A bunch of stuff, mainly two categories:
            1. values that we actually care, like the ones updated in the
               current iterations and to be used in the next one
               a. flatten_position
               b. flatten_walker_age
               c. flatten_weight,
               d. key,
               e. energy_offset,
               f. mixed_estimator (This is the energy estimator that we care at the
                  end of day)
            2. debug info. (all other returned values)
        '''
        key, subkey = jax.random.split(key)

        if state.walker_age is not None:
            assert state.position.shape[0] == state.walker_age.shape[0]

        if ebye_move:
            (flatten_position, flatten_energy, flatten_weight_delta_log, acceptance_rate, effective_time_step_list, debug_info) = dmc_step(
                state.position, state.walker_age, state.local_energy, subkey,
                state.energy_offset, state.mixed_estimator)
            flatten_local_energy = flatten_energy
            acceptance_ratio = agg_mean(acceptance_rate)

            # We use the strategy which CASINO recommend, that is, when we update walkers' weights,
            # the effective time step is chosen to be <p*delta_R^2>/<delta_R^2> of current iteration
            # for each walker. Note that p is electron acceptance rate and walkers' effective time
            # step values are different, which is why we have a effective_time_step_list.
            flatten_weight = state.weight * jnp.exp(effective_time_step_list * flatten_weight_delta_log)

            # When we calculate mixed energy and update energy offset, we use accumulated effective time step.
            if should_update_effective_time_step:
                state.effective_time_step_calculator.update_ebye(
                    effective_time_step_list = effective_time_step_list,
                    weights=flatten_weight)
            effective_time_step = state.effective_time_step_calculator.run()

        else:
            (flatten_position, flatten_energy, flatten_walker_age, flatten_local_energy,
            flatten_weight_delta_log, delta_R, acceptance_rate,
            debug_info) = dmc_step(
                state.position, state.walker_age, state.local_energy, subkey,
                state.energy_offset, state.mixed_estimator)

            if should_update_effective_time_step:
                state.effective_time_step_calculator.update(
                    diffusion_displacement=delta_R,
                    acceptance_rate=acceptance_rate,
                    weights=state.weight)
            effective_time_step = state.effective_time_step_calculator.run()
            acceptance_ratio = agg_mean(acceptance_rate)
            flatten_weight = state.weight * jnp.exp(effective_time_step * flatten_weight_delta_log)

        # We check outlier before branching to avoid meaningless branching work
        # in the case of outlier.
        averaged_energy_before_branch = average_energy(flatten_energy, flatten_weight, energy_clip_pair)
        key, subkey = jax.random.split(key)

        if is_energy_outlier(averaged_energy_before_branch, state.mixed_estimator):
            # If the current run produce outrageous energy, we skip it and start
            # over in the next run (wishing for better luck)
            logging.warning(f'Hitting energy outlier {averaged_energy_before_branch}, anchor: {state.mixed_estimator}')
            # we return a new key wishing for better luck
            # And we don't advance the index.
            return index, IterationOutput(
                    succeeded=False,
                    state=state,
                    key=key)

        total_weight_before_branch = agg_sum(flatten_weight)

        if ebye_move:
            # We don't track walker age in ebye mode.
            branch_arrays = (flatten_position, flatten_energy, flatten_local_energy)
        else:
            branch_arrays = (flatten_position, flatten_energy, flatten_walker_age, flatten_local_energy)

        flatten_weight, branch_arrays = branch(
            flatten_weight,
            subkey,
            branch_arrays,
            min_thres=weight_branch_threshold[0],
            max_thres=weight_branch_threshold[1])

        if ebye_move:
            flatten_position, flatten_energy, flatten_local_energy = branch_arrays
            flatten_walker_age = None
            num_old_walkers = 0
        else:
            flatten_position, flatten_energy, flatten_walker_age, flatten_local_energy = branch_arrays
            num_old_walkers = agg_sum(flatten_walker_age > 20)
        assert flatten_position.shape[0] == flatten_weight.shape[0]

        total_weight = agg_sum(flatten_weight)
        averaged_energy = average_energy(flatten_energy, flatten_weight, energy_clip_pair)

        try:
            np.testing.assert_array_almost_equal_nulp(total_weight_before_branch, total_weight)
        except Exception:
            logging.warning(f'before branch weight: {total_weight_before_branch}, afterwards: {total_weight}')

        mixed_estimator = state.mixed_estimator_calculator.run(
            state.energy_offset,
            averaged_energy,
            total_weight,
            time_step=effective_time_step)

        updated_energy_offset_update_amplitude = energy_offset_update_amplitude *  time_step / effective_time_step
        # The energy offset at time t is calculated using the energy estimator
        # at time (t - 1), then it's used to calculate the energy estimator at time t.
        if index % update_energy_offset_interval == 0:
            energy_offset = update_energy_offset(
                energy_estimator=mixed_estimator,
                amplitude=updated_energy_offset_update_amplitude,
                curr_total_weight=total_weight,
                target_weight=state.target_num_walkers)
        else:
            energy_offset = state.energy_offset

        new_state = State(
            position=flatten_position,
            walker_age=flatten_walker_age,
            weight=flatten_weight,
            local_energy=flatten_local_energy,
            energy_offset=energy_offset,
            target_num_walkers=state.target_num_walkers,
            mixed_estimator=mixed_estimator,
            mixed_estimator_calculator=state.mixed_estimator_calculator,
            effective_time_step_calculator=state.effective_time_step_calculator)
        new_index = index + 1

        return new_index, IterationOutput(
            succeeded=True,
            state=new_state,
            key=key,
            average_energy=averaged_energy,
            num_old_walkers=num_old_walkers,
            acceptance_ratio=acceptance_ratio,
            effective_time_step=effective_time_step,
            debug_info=debug_info)
    return dmc_single_iteration

def average_energy(energy, weight, energy_clip_pair):
    '''
    calculate the weighted and clipped average of local energy.
    The input weight is not necessarily normalized.
    '''
    if energy_clip_pair is not None:
        clipped_energy = jnp.clip(energy, energy_clip_pair[0], energy_clip_pair[1])
    else:
        clipped_energy = energy
    normed_weight = weight / agg_sum(weight)
    return agg_sum(clipped_energy * normed_weight)

@jax.jit
def update_energy_offset(energy_estimator, amplitude, curr_total_weight, target_weight):
    '''
    If `curr_total_weight` is less than `target_weight`, we want to have
    larger energy offset so that more walkers can be born, otherwise we set
    energy offset to be small so that more walkers will be killed.

    Args:
        energy_estimator: The current energy estimator
        amplitude: how strongly you want to pull the number of walkers to the target one
        curr_total_weight: The current total weight of all walkers.
        target_weight: The target total weight of all walkers.
    '''
    return energy_estimator - amplitude * jnp.log(curr_total_weight / target_weight)

def calc_branch_val(velocity, clipped_velocity, local_energy,
                    time_step, energy_offset, mixed_estimator,
                    energy_cutoff_alpha=0.2, energy_clip_pair=None):
    # We need to do some clipping here to prevent outliers to have super
    # large weight.
    if energy_clip_pair is not None:
        local_energy = jnp.clip(local_energy, energy_clip_pair[0], energy_clip_pair[1])

    if energy_cutoff_alpha < 0:
        # fall back to UNR algo.
        energy_diff = (
            energy_offset - mixed_estimator
            + (mixed_estimator - local_energy) * (jnp.linalg.norm(clipped_velocity) / jnp.linalg.norm(velocity)))
        return energy_diff, False

    num_electrons = len(velocity) // 3
    energy_cutoff = energy_cutoff_alpha * jnp.sqrt(num_electrons / time_step)

    energy_diff = (
        energy_offset - mixed_estimator
        + jnp.sign(mixed_estimator - local_energy) * jnp.minimum(energy_cutoff, jnp.abs(mixed_estimator - local_energy)))
    return energy_diff, energy_cutoff < jnp.abs(mixed_estimator - local_energy)

def do_run_dmc_single_walker(position, walker_age, local_energy,
                             vmc_wave_func_with_sign, velocity_func, local_energy_func,
                             time_step, key,
                             energy_offset,
                             mixed_estimator,
                             nuclei,
                             charges,
                             energy_cutoff_alpha=0.2,
                             energy_clip_pair=None):
    '''
    Handling position for a single walker
    Return:
        1. new position of the walker (dim-3N)
        2. weighted average between the old walker local energy and the new walker local energy(scalar), to be used in the calculation of final averaged energy (scalar)
        3. age of the new walker (scalar)
        4. local energy of the new walker (scalar)
        5. log of delta weight of the new walker except the effective time step term (scalar)
        6. displacement of diffusion (scalar)
        7. acceptance_rate of the new walker (scalar)
        4 - 7 are returned to compute the effective time step, which is not easy
        to calculate at this level.
        8. a bunch of debug info
    '''
    velocity = velocity_func(position)

    key, subkey = jax.random.split(key)
    updated_position, clipped_velocity, G_log, delta_R = from_orig_config_wrapper(
        position,
        velocity,
        nuclei,
        charges,
        time_step,
        subkey)

    updated_velocity = velocity_func(updated_position)
    key, subkey = jax.random.split(key)
    _, clipped_updated_velocity, G_log_from_updated = from_updated_config_wrapper(
        position,
        updated_position,
        updated_velocity,
        nuclei,
        charges,
        time_step,
        subkey)
    wf_sign, wf_val = vmc_wave_func_with_sign(position)
    wf_sign_diffused, wf_val_diffused = vmc_wave_func_with_sign(updated_position)

    acceptance_rate_log = (
        2 * (wf_val_diffused - wf_val)
        + G_log_from_updated - G_log
        + jnp.log(1.1) * jnp.clip(walker_age - 50, 0, jnp.inf))
    acceptance_rate = jax.lax.cond(
        wf_sign * wf_sign_diffused < 0,
        lambda _: 0.0,
        lambda _: jnp.clip(jnp.exp(acceptance_rate_log), 0.0, 1.0),
        operand=None)

    local_energy_updated = local_energy_func(updated_position)
    branch_val_from_orig, cutoff_bool_from_orig = calc_branch_val(
        velocity, clipped_velocity, local_energy,
        time_step, energy_offset, mixed_estimator,
        energy_cutoff_alpha=energy_cutoff_alpha, energy_clip_pair=energy_clip_pair
    )
    branch_val_from_updated, cutoff_bool_from_updated = calc_branch_val(
        updated_velocity, clipped_updated_velocity, local_energy_updated,
        time_step, energy_offset, mixed_estimator,
        energy_cutoff_alpha=energy_cutoff_alpha, energy_clip_pair=energy_clip_pair
    )
    weight_delta_log = (
        acceptance_rate / 2 * (branch_val_from_orig + branch_val_from_updated)
        + (1 - acceptance_rate) * branch_val_from_orig)

    uniform_var_for_acceptance = jax.random.uniform(key)
    is_accepted = uniform_var_for_acceptance < acceptance_rate

    average_local_energy_new = acceptance_rate * local_energy_updated + (1 - acceptance_rate) * local_energy
    position_new, walker_age_new, local_energy_new = jax.lax.cond(
        is_accepted,
        lambda _: (updated_position, 1.0, local_energy_updated),
        lambda _: (position, walker_age + 1, local_energy),
        operand=None)

    return (position_new,
            average_local_energy_new,
            walker_age_new,
            local_energy_new,
            weight_delta_log,
            delta_R,
            acceptance_rate,
            (is_accepted, acceptance_rate,
             local_energy, local_energy_updated,
             jnp.linalg.norm(velocity, axis=-1),
             jnp.linalg.norm(updated_velocity, axis=-1),
             cutoff_bool_from_orig,
             cutoff_bool_from_updated
            )
           )

def do_run_dmc_single_walker_ebye(position, walker_age, local_energy,
                             vmc_wave_func_with_sign, velocity_func, local_energy_func,
                             time_step, key,
                             energy_offset,
                             mixed_estimator,
                             nuclei,
                             charges,
                             energy_cutoff_alpha=0.2,
                             energy_clip_pair=None):
    '''
    Handling position for a single walker, elec-by-elec moves.
    Return:
        1. new position of the walker (dim-3N)
        2. local energy of the new walker (scalar)
        3. log of delta weight of the new walker except the effective time step term (scalar)
        4. effective time step (scalar)
        5. a bunch of debug info
    '''
    del walker_age

    velocity = velocity_func(position)

    key, subkey = jax.random.split(key)
    updated_position, clipped_velocity, updated_velocity, clipped_updated_velocity, acceptance_rate, effective_time_step = from_orig_config_wrapper_ebye(
        position,
        velocity,
        nuclei,
        charges,
        vmc_wave_func_with_sign,
        velocity_func,
        time_step,
        subkey)

    local_energy_updated = local_energy_func(updated_position)
    branch_val_from_orig, cutoff_bool_from_orig = calc_branch_val(
        velocity, clipped_velocity, local_energy,
        time_step, energy_offset, mixed_estimator,
        energy_cutoff_alpha=energy_cutoff_alpha, energy_clip_pair=energy_clip_pair
    )
    branch_val_from_updated, cutoff_bool_from_updated = calc_branch_val(
        updated_velocity, clipped_updated_velocity, local_energy_updated,
        time_step, energy_offset, mixed_estimator,
        energy_cutoff_alpha=energy_cutoff_alpha, energy_clip_pair=energy_clip_pair
    )
    weight_delta_log = (1 / 2 * (branch_val_from_orig + branch_val_from_updated))

    debug_info = (
        local_energy,
        local_energy_updated,
        jnp.linalg.norm(velocity, axis=-1),
        jnp.linalg.norm(updated_velocity, axis=-1),
        cutoff_bool_from_orig,
        cutoff_bool_from_updated
    )

    return (updated_position,
            local_energy_updated,
            weight_delta_log,
            acceptance_rate,
            effective_time_step,
            debug_info
           )

def run_dmc_single_walker(ebye_move, mask, *args, **kwargs):
    '''
    Handle mask. Return dummy value if the mask is zero
    See `do_run_dmc_single_walker` for info on argument and return value.
    This method is expected to be pmapped, so no need to be jit'ed
    '''
    if ebye_move:
        *result, debug_info = do_run_dmc_single_walker_ebye(*args, **kwargs)
    else:
        *result, debug_info = do_run_dmc_single_walker(*args, **kwargs)
    # Here we do multiplication with mask instead of using `lax.cond` due to
    # performance concern, that when using `lax.cond` we experienced a 4~5 time
    # decrement of efficiency.
    return [jnp.nan_to_num(a) * mask for a in result] + [debug_info]


def get_to_pad_num(num_walkers, num_device):
    least_target = num_walkers // num_device + 1
    num_digit = int(np.log10(least_target))
    top_2_digit = least_target // (10 ** (num_digit - 1))
    target_num = (top_2_digit + 1) * (10 ** (num_digit - 1))
    return int(target_num * num_device - num_walkers)

def make_step(vmc_wave_func_with_sign, velocity_func, local_energy_func,
              time_step,
              nuclei,
              charges,
              energy_cutoff_alpha,
              energy_clip_pair=None,
              do_padding=True,
              ebye_move=False):
    '''
    The "step" here is responsible for updating walkers' positions and weights with pmap
    The wrapper handle the following logic:
        1. Do pmap (Due to the fact that pmap will do jit automatically, we don't
           need to do jit manually)
        2. Do padding and reshape to make pmap happy. Also split key so that each
           device use different RNG key.
        3. Do mask to handle the padded dummy value.

    To handle the case where the number of walkers cannot be divided by the number
    of devices, we needs to do padding. However, doing padding and removing the padded
    elements is not for free and could be noticeable especially when we are dealing
    with fast trial functions, for instance, a very small FermiNet.
    If the caller can guarantee that no padding is needed, he/she can pass in
    `do_padding=False`, in which case no padding procedure would be invoked.
    '''
    num_device = jax.local_device_count()
    run_dmc_single_walker_closure = lambda mask, z, age, local_energy, key, energy_offset, mixed_estimator: run_dmc_single_walker(
        ebye_move, mask, z, age, local_energy,
        vmc_wave_func_with_sign, velocity_func, local_energy_func,
        time_step,
        key,
        energy_offset,
        mixed_estimator,
        nuclei,
        charges,
        energy_cutoff_alpha=energy_cutoff_alpha,
        energy_clip_pair=energy_clip_pair)

    # position, walker_age, weight, mask, key and random key are all vectorized.
    # Energy_offset and mixed_estimator are not
    in_axes = (0, 0, 0, 0, 0, None, None)
    jitted_vmapped_func = jax.jit(jax.vmap(run_dmc_single_walker_closure, in_axes=in_axes))
    pmaped_func = jax.pmap(jitted_vmapped_func, in_axes=in_axes)

    local_energy_jitted_vmapped_func = jax.jit(jax.vmap(local_energy_func))
    local_energy_pmaped_func = jax.pmap(local_energy_jitted_vmapped_func)


    def run(flatten_position, flatten_walker_age, flatten_local_energy, key,
            energy_offset, mixed_estimator):
        '''
        See docstrings of `make_dmc_single_iteration` for more information on the
        arguments.
        See docstrings of `do_run_dmc_single_walker` for more info on the return values.
        (Mainly the position, local energy, age and weight of the new walkers, and some
         debug info)
        '''
        num_walkers, walker_dim = flatten_position.shape

        to_pad_num = get_to_pad_num(num_walkers, num_device) if do_padding else 0
        def do_pad_and_reshape(array):
            if not do_padding:
                return array.reshape((num_device, -1))
            return jnp.pad(
                array,
                ((0, to_pad_num),),
                constant_values=0.0).reshape((num_device, -1))

        mask = do_pad_and_reshape(jnp.ones(num_walkers))
        # For algo like ebye move, we don't track walker age, in which case
        # `flatten_walker_age` is None.
        if flatten_walker_age is not None:
            walker_age = do_pad_and_reshape(flatten_walker_age)
        else:
            walker_age = None

        if not do_padding:
            position = flatten_position.reshape((num_device, -1, walker_dim))
        else:
            position = jnp.pad(
                flatten_position,
                ((0, to_pad_num), (0, 0)),
                constant_values=0).reshape((num_device, -1, walker_dim))

        target_shape = position.shape[:2]
        split_keys = jax.random.split(key, target_shape[0] * target_shape[1])
        reshaped_keys = split_keys.reshape(tuple(target_shape) + (-1, ))

        if flatten_local_energy is None:
            local_energy = local_energy_pmaped_func(position)
        else:
            local_energy = do_pad_and_reshape(flatten_local_energy)

        *result, debug_info = pmaped_func(mask, position, walker_age, local_energy, reshaped_keys, energy_offset, mixed_estimator)
        # Remove fake-masked elements. Otherwise they makes troubles
        # when spliting/merging walkers.
        # BTW, the repeat operation here prevent the function from being jit'ed, because the
        # output shape is unknown statically afterwards
        if do_padding:
            flatten_mask = flatten(mask).astype(int)
            return [flatten(r).repeat(flatten_mask, axis=0) for r in result] + [debug_info]
        return [flatten(r) for r in result] + [debug_info]

    # If no padding is done, we can do JIT, though not much efficiency improvement.
    # Moveover, JITing in that case would trigger the following warning:
    # """
    # UserWarning: The jitted function run includes a pmap. Using jit-of-pmap can lead to inefficient data movement,
    # as the outer jit does not preserve sharded data representations and instead collects input and output arrays onto a single device.
    # Consider removing the outer jit unless you know what you're doing. See https://github.com/google/jax/issues/2926
    # """
    return run

def flatten(arr):
    '''
    Only handle shape == 2 or 3 arrays.
    '''
    if len(arr.shape) == 2:
        return arr.reshape((-1, ))
    if len(arr.shape) == 3:
        return arr.reshape((-1, arr.shape[-1]))
    raise Exception(f'Unexpected shape: {arr.shape}')

def recovery_wrapper(#dmc_single_iteration: DMC_ITERATION,
        dmc_single_iteration,
        max_restore_nums,
        ckpt_metric_manager,
        key):
    curr_restore_nums = 0
    nonlocal_key = key
    def helper(*args, **kwargs):
        nonlocal curr_restore_nums
        nonlocal nonlocal_key
        try:
            return dmc_single_iteration(*args, **kwargs)
        except Exception as e:
            curr_restore_nums += 1
            logging.warning(f'Failed try due to {e}')
            if curr_restore_nums > max_restore_nums:
                logging.warning('Exceeding max restore number. Aborting...')
                raise e
            # May raise `NoSafeDataAvailable` if not safe data is available
            # for restoring.
            nonlocal_key, step, restored_data = restore_and_rollback_dmc_data(ckpt_metric_manager, nonlocal_key)
            logging.info(f'Recovered state at step {step}')
            return step + 1, restored_data
    return helper

def restore_and_rollback_dmc_data(ckpt_metric_manager: CkptMetricManager,
                          key: jax.random.PRNGKey):
    step, state = ckpt_metric_manager.restore_and_rollback()
    for _ in range(66):
        key, subkey = jax.random.split(key)
    return key, step, IterationOutput(
            succeeded=False,
            state=state,
            key=subkey)
