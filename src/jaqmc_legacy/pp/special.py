# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp


def legendre(x, l):
    if l == 0:
        return jnp.ones_like(x)
    if l == 1:
        return x
    if l == 2:
        return (3 * x**2 - 1) / 2
    # add more legendre polynomials for l = {3,4,5,6,7} if needed
    # for truncation order more than 3
    if l == 3:
        return (5 * x**3 - 3 * x) / 2
    if l == 4:
        return (35 * x**4 - 30 * x**2 + 3) / 8
    if l == 5:
        return (63 * x**5 - 70 * x**3 + 15 * x) / 8
    if l == 6:
        return (231 * x**6 - 315 * x**4 + 105 * x**2 - 5) / 16
    if l == 7:
        return (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x) / 16
    else:
        pass

def legendre_list(x, l_list):
    result = []
    for l in l_list:
        result.append(legendre(x, l)[None, ...])
    return jnp.concatenate(result, axis=0)
