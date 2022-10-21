# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Variable-size branching.
'''

from functools import partial

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

def branch(weight, key, branch_arrays, min_thres=0.3, max_thres=2):
    '''
    Split large weight, merge small weight.
    We will only be able to do branching within the same host, no balancing
    across hosts for now

    Args:
        weight: The weight according to which the split and merge will be done.
        key: The RNG key used to determine which walker should be kept when merging
             two walkers.
        branch_arrays: A list of arrays with the same length as `weight`. Those
                       arrays will be branched accordingly.
        min_thres, max_thres: The threshold used to determine which walkers to
                              split and merge.
    '''
    merge_pairs = round_merge_pairs(int(jnp.sum(weight < min_thres)) // 2 + 1)
    if merge_pairs > 0.05 * len(weight):
        logging.warning(f'large number of pairs to merge: {merge_pairs}')
    weight, repeat_num = do_branch(weight, key, merge_pairs,
                                   min_thres=min_thres,
                                   max_thres=max_thres)
    weight = weight.repeat(repeat_num, axis=0)
    return weight, [l.repeat(repeat_num, axis=0) for l in branch_arrays]

def round_merge_pairs(target_num):
    '''
    In order to reduce the number of re-compilation of `do_branch` method, we
    round the `merge_pairs` value. (Note that `do_branch` will be re-compiled every
    time it meets a new value of `merge_pairs`.)
    '''
    if target_num <= 10:
        return target_num
    num_digit = int(np.log10(target_num))
    most_sig_digit = target_num // (10 ** num_digit)
    return most_sig_digit * (10 ** num_digit)

@partial(jax.jit, static_argnums=(2,))
def do_branch(weight, key, merge_pairs, min_thres=0.1, max_thres=1):
    repeat_num = jnp.ones(weight.shape, dtype='int32')

    # We take the k-smallest value of weight and their indices by first multiplying
    # the weight array by -1 then take the top-k elements.
    neg_smallest_k, smallest_k_indices = jax.lax.top_k(-1 * weight, min(2 * merge_pairs, len(weight)))
    smallest_k = -neg_smallest_k.reshape((merge_pairs, 2))
    smallest_k_indices = smallest_k_indices.reshape((merge_pairs, 2))

    def helper(index, values):
        weight, repeat_num, key = values
        k1, k2 = smallest_k[index]
        k1_index, k2_index = smallest_k_indices[index]

        weight, repeat_num, key = jax.lax.cond(
            k1 < min_thres,
            update_weight,
            lambda x: x[0],
            ((weight, repeat_num, key), k1, k2, k1_index, k2_index))
        return weight, repeat_num, key

    weight, repeat_num, key = jax.lax.fori_loop(
        0,
        merge_pairs,
        helper,
        (weight, repeat_num, key))

    repeat_num *= 1 + (weight > max_thres)
    weight *= 1 - (weight > max_thres) / 2

    return weight, repeat_num

def update_weight(_input):
    (weight, repeat_num, key), k1, k2, k1_index, k2_index = _input
    key, sub_key = jax.random.split(key)
    keep_index, rm_index = jax.lax.cond(jax.random.uniform(sub_key) < k1 / (k1 + k2),
                                        lambda _: (k1_index, k2_index),
                                        lambda _: (k2_index, k1_index),
                                        operand=None)
    weight = weight.at[keep_index].set(k1 + k2)
    repeat_num = repeat_num.at[rm_index].set(0)
    return weight, repeat_num, key
