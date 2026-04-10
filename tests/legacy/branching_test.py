# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import time

import chex
import jax
import pytest
from jax import numpy as jnp

from jaqmc_legacy.dmc.branch import do_branch, round_merge_pairs
from jaqmc_legacy.dmc.branch_fix_size import branch as branch_fix_size


def test_branch_do_branch() -> None:
    key = jax.random.PRNGKey(42)
    weight = jnp.array([0.01, 0.02, 0.1, 0.1, 0.2, 0.3, 0.44, 1.6, 1.8, 3, 4])

    updated_weight, repeat_num = do_branch(
        weight,
        key,
        5,
        min_thres=0.5,
        max_thres=2,
    )

    expected_weight = jnp.array(
        [0.01, 0.03, 0.2, 0.1, 0.5, 0.3, 0.44, 1.02, 1.8, 1.5, 2]
    )
    expected_repeat_num = jnp.array([0, 1, 1, 0, 1, 0, 0, 2, 1, 2, 2])

    assert float(jnp.sum(updated_weight * repeat_num)) == pytest.approx(
        float(jnp.sum(weight)),
        rel=1e-5,
    )
    chex.assert_trees_all_close(updated_weight, expected_weight)
    chex.assert_trees_all_close(repeat_num, expected_repeat_num)


def test_round_merge_pairs_small_is_identity() -> None:
    for num in range(11):
        assert round_merge_pairs(num) == num


@pytest.mark.parametrize(
    ("num", "expected"),
    [(25, 20), (35, 30), (120, 100), (1200, 1000), (2200, 2000), (9200, 9000)],
)
def test_round_merge_pairs_rounding(num: int, expected: int) -> None:
    assert round_merge_pairs(num) == expected


def test_branch_fix_size_branch() -> None:
    test_weight = jnp.array([0.0, 0.1, 0.5, 0.5, 4.0])
    key = jax.random.PRNGKey(int(1e6 * time.time()))
    before_branch_array = jnp.array([1, 2, 3, 4, 5])

    expected_after_branch_array = jnp.array([5, 2, 3, 4, 5])
    expected_weight = jnp.array([2.0, 0.1, 0.5, 0.5, 2.0])

    for test_min_thres, test_max_thres in [(0.3, 2.0), (-1.0, 2.0), (0.2, 5.0)]:
        actual_weight, [actual_branch_array] = branch_fix_size(
            test_weight,
            key,
            [before_branch_array],
            min_thres=test_min_thres,
            max_thres=test_max_thres,
        )
        chex.assert_trees_all_close(actual_weight, expected_weight)
        chex.assert_trees_all_close(actual_branch_array, expected_after_branch_array)
