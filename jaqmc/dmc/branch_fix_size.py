# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
fix-size branching.
'''

from absl import logging
import jax
import jax.numpy as jnp

def branch(weight, key, branch_arrays, min_thres=0.3, max_thres=2):
    '''
    Split large weight, merge small weight.
    We will only be able to do branching within the same host, no balancing
    across hosts for now

    We apply a simple trick to keep the population size fixed:
    Whenever we want to merge a pair, we also split a walker (with largest weight);
    similarly whenever  we want to split a walker, we merge a pair(with smallest weight).

    This trick should be harmless with reasonable branching thresholds. That said,
    one potential issue is that we merge two not-so-small-weight walkers and generate
    one large-weight walker whose weight may exceed the `max_thres`. This may
    happen when `max_thres` is too small while `min_thres` is too large (for instance,
    if we choose the thresholds as (0.8, 1.2)). If our thresholds are reasonble, (`min_thres`
    0.2~0.4, `max_thres` 1.8 ~ 2.0), then together with assumption that the majority
    of walkers have weight around 1, we should be fine.

    Args:
        weight: The weight according to which the split and merge will be done.
        key: The RNG key used to determine which walker should be kept when merging
             two walkers.
        branch_arrays: A list of arrays with the same length as `weight`. Those
                       arrays will be branched accordingly.
        min_thres, max_thres: The threshold used to determine which walkers to
                              split and merge.
    '''
    # Round upwards. Namely whenever we have a weight below the threshold, we
    # merge it with the second smallest one. It should not matter much compared
    # to the more conservative approach doing downward rounding.
    num_merge_pairs = (jnp.sum(weight < min_thres) + 1) // 2
    num_split_walkers = jnp.sum(weight > max_thres)
    # We will merge and split the same number of small-weight pairs and large-weight
    # walkers, so that the population size is not changed.
    num_to_change = jnp.maximum(num_merge_pairs, num_split_walkers).tolist()
    if num_to_change > 0.05 * len(weight):
        logging.warning(f'large number to change: {num_to_change}')
    if num_to_change == 0:
        return weight, branch_arrays

    # We take the k-smallest value of weight and their indices by first multiplying
    # the weight array by -1 then take the top-k elements.
    _, smallest_k_indices = jax.lax.top_k(-1 * weight, 2 * num_to_change)

    # Group the smallest indices to pairs to be merged.
    smallest_k_indices = smallest_k_indices.reshape((num_to_change, 2))

    _, largest_k_indices = jax.lax.top_k(weight, num_to_change)
    weight, *branch_arrays = do_branch(weight, key, smallest_k_indices, largest_k_indices, *branch_arrays)

    return weight, branch_arrays

@jax.jit
def do_branch(weight, key, smallest_k_indices, largest_k_indices, *branch_arrays):
    thresholds = weight[smallest_k_indices[:, 0]] / (weight[smallest_k_indices[:, 0]] + weight[smallest_k_indices[:, 1]])
    random_num = jax.random.uniform(key, shape=thresholds.shape)
    kept_indices = jnp.where(random_num < thresholds, smallest_k_indices[:, 0], smallest_k_indices[:, 1])
    removed_indices = jnp.where(random_num > thresholds, smallest_k_indices[:, 0], smallest_k_indices[:, 1])

    # For arrays in `branch_arrays`, the spots for removed elements will be simply filled with the branched ones.
    branch_arrays = [arr.at[removed_indices].set(arr[largest_k_indices]) for arr in branch_arrays]

    # For weights, it's trickier. We need to add the removed elements' weight to the
    # winner in the merging process. And then halve the weight of the branched elements,
    # then copy it to the spots for the removed elements.
    weight = weight.at[kept_indices].add(weight[removed_indices])
    weight = weight.at[largest_k_indices].set(weight[largest_k_indices] / 2)
    weight = weight.at[removed_indices].set(weight[largest_k_indices])
    return [weight] + branch_arrays
