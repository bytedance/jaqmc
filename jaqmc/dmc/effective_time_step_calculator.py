# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Effective time step calculation
'''

import jax.numpy as jnp

from .utils import agg_mean, agg_sum

class EffectiveTimeStepCalculator:

    def __init__(self, time_step):
        self.total_weight = 0.0
        self.denominator = 0.0
        self.numerator = 0.0
        self.time_step = time_step

    def update(self,
               diffusion_displacement,
               acceptance_rate,
               weights):
        old_total_weight = self.total_weight
        total_of_new_weights = agg_sum(weights)
        self.total_weight += total_of_new_weights

        new_avg_diffusion = agg_mean(diffusion_displacement ** 2,
                                     weights=weights)
        new_avg_accepted_diffusion = agg_mean(acceptance_rate * (diffusion_displacement ** 2),
                                              weights=weights)

        self.denominator = (
            self.denominator * (old_total_weight / self.total_weight)
            + new_avg_diffusion * (total_of_new_weights / self.total_weight))# avg_diffusion_on_denominator
        self.numerator = (
            self.numerator * (old_total_weight / self.total_weight)
            + new_avg_accepted_diffusion * (total_of_new_weights / self.total_weight))# avg_accepted_diffusion_on_numerator

    def update_ebye(self,
                    effective_time_step_list,
                    weights):
        total_of_new_weights = agg_sum(weights)
        self.total_weight += total_of_new_weights

        new_sum_effective_time_step = agg_sum(effective_time_step_list * weights)

        self.denominator = self.total_weight * self.time_step
        self.numerator += new_sum_effective_time_step

    def run(self):
        if self.total_weight == 0.0 or self.denominator == 0.0:
            return self.time_step
        return self.time_step * jnp.exp(jnp.log(self.numerator) - jnp.log(self.denominator))
