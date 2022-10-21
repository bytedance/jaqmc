# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Misc util functions.
'''

from functools import partial

import jax
import jax.numpy as jnp

compute_mean = jax.pmap(lambda x: jax.lax.pmean(x, "i"), axis_name="i")
compute_sum = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")

def agg_helper(x, p_agg_func, local_agg_func):
    '''
    Do local aggregation with `local_agg_func` (like `jnp.sum`) then do global aggregation
    with `p_agg_func` (like `compute_sum`)
    '''
    # Handle scalar case
    x = jnp.asarray(x)

    # Another way of implementation is to reshape the input `x` to a pmap'able
    # shape. However that requires the array length to be divisible by the number
    # of (local) devices, which may not be satisfied in DMC.
    local_agg_result = jnp.ones(jax.local_device_count()) * local_agg_func(x)
    return p_agg_func(local_agg_result)[0]

_agg_mean = partial(agg_helper, p_agg_func=compute_mean, local_agg_func=jnp.mean)
_agg_sum = partial(agg_helper, p_agg_func=compute_sum, local_agg_func=jnp.sum)

# Idealy we should just do `jnp.average` to avoid overhead here for single-host case.
# However we didn't find a neat way to do such logic branching. Will leave it for future work
# TODO Do `jnp.average` for single-host case and do `agg_mean` for multiple-host case.
def agg_mean(x, weights=None):
    '''
    Do global and across-hosts `jnp.average`
    '''
    if weights is None:
        return _agg_mean(x)
    total_weight = agg_sum(weights)
    weighted_x = x * weights / total_weight
    return agg_sum(weighted_x)

def agg_sum(x):
    '''
    Do global and across-hosts `jnp.sum`
    '''
    # `_agg_sum` would sum over more elements (with multiple `jax.local_device_count()`)
    double_counted_result = _agg_sum(x)
    return double_counted_result / jax.local_device_count()
