# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Data classes representing the internal states of DMC.
'''

from typing import Optional

import attr
import jax
import jax.numpy as jnp
import numpy as np

from .energy_estimator import MixedEstimatorCalculator
from .effective_time_step_calculator import EffectiveTimeStepCalculator
from .utils import agg_sum

@attr.s(auto_attribs=True)
class State:
    # Position, walker_age, weight, local_energy are
    # expected to be flattened, in the sense the size of their first dimension
    # is the total number of the walkers in this batch.
    position: Optional[jnp.ndarray] = None
    walker_age: Optional[jnp.ndarray] = None
    weight: Optional[jnp.ndarray] = None
    local_energy: Optional[jnp.ndarray] = None
    energy_offset: Optional[float] = None
    target_num_walkers: Optional[int] = None
    mixed_estimator: Optional[float] = None
    mixed_estimator_calculator: Optional[MixedEstimatorCalculator] = None
    effective_time_step_calculator: Optional[EffectiveTimeStepCalculator] = None

    @classmethod
    def default(cls,
                init_position: Optional[jnp.ndarray],
                calc_energy_func,
                mixed_estimator_num_steps,
                energy_window_size,
                time_step):
        '''
        Create a default State, for instance, for the initial step of DMC.
        '''
        default_walker_age = np.ones(len(init_position))
        default_weight = np.ones(len(init_position))
        default_energy = calc_energy_func(init_position)
        default_target_num_walkers = agg_sum(len(init_position))
        default_mixed_estimator_calculator = MixedEstimatorCalculator(
            mixed_estimator_num_steps=mixed_estimator_num_steps,
            energy_window_size=energy_window_size)
        default_effective_time_step_calculator = EffectiveTimeStepCalculator(time_step)

        return cls(
            position=init_position,
            walker_age=default_walker_age,
            weight=default_weight,
            local_energy=None,
            energy_offset=default_energy,
            target_num_walkers=default_target_num_walkers,
            mixed_estimator=default_energy,
            mixed_estimator_calculator=default_mixed_estimator_calculator,
            effective_time_step_calculator=default_effective_time_step_calculator)

@attr.s(auto_attribs=True)
class IterationOutput:
    '''
    The output of a DMC iteration.

    A ckpt file should be able to recover all the data listed so that the process
    can be continued.
    '''
    succeeded: bool
    state: State
    key: jax.random.PRNGKey
    average_energy: Optional[float] = None
    num_old_walkers: Optional[int] = None
    acceptance_ratio: Optional[float] = None
    effective_time_step: Optional[float] = None
    debug_info: Optional[dict] = None
