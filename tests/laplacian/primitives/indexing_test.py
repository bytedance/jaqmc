# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Indexing primitive semantics (gather/dynamic_slice/scatter*)."""

import operator

import jax
import jax.numpy as jnp
import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import random_array, tracked_case_input

from .helpers import check_unary, parametrize_over_tracked_cases


def _scatter_add(v):
    out = jnp.zeros((5, *v.shape[1:]), dtype=v.dtype)
    return out.at[jnp.array([0, 2, 4])].add(v**2)


def _scatter_set(v):
    out = jnp.zeros((5, *v.shape[1:]), dtype=v.dtype)
    return out.at[jnp.array([0, 2, 4])].set(v**2)


def _scatter_max(v):
    out = jnp.full((5, *v.shape[1:]), -10.0, dtype=v.dtype)
    return out.at[jnp.array([0, 2, 4])].max(v)


def _scatter_min(v):
    out = jnp.full((5, *v.shape[1:]), 10.0, dtype=v.dtype)
    return out.at[jnp.array([0, 2, 4])].min(v)


def _scatter_add_duplicate_indices(v):
    out = jnp.zeros((5, *v.shape[1:]), dtype=v.dtype)
    return out.at[jnp.array([0, 0, 2])].add(v**2)


def _scatter_set_duplicate_indices(v):
    out = jnp.zeros((5, *v.shape[1:]), dtype=v.dtype)
    return out.at[jnp.array([1, 1, 3])].set(v**2)


def _scatter_max_base_wins(v):
    out = jnp.full((5, *v.shape[1:]), 100.0, dtype=v.dtype)
    return out.at[jnp.array([0, 2, 4])].max(v)


def _scatter_min_base_wins(v):
    out = jnp.full((5, *v.shape[1:]), -100.0, dtype=v.dtype)
    return out.at[jnp.array([0, 2, 4])].min(v)


def _dynamic_slice_with_start_arg(v, start):
    return jax.lax.dynamic_slice(v, (start, 0), (2, v.shape[1]))


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(operator.itemgetter(jnp.array([0, 2])), "real", id="gather_rows"),
        pytest.param(
            lambda v: jnp.take(v, jnp.array([2, 0, 1]), axis=-1),
            "real",
            id="gather_last_axis",
        ),
        pytest.param(
            lambda v: jnp.take_along_axis(
                v,
                jnp.array(
                    [[0, 2, 1], [2, 0, 1], [1, 2, 0]],
                    dtype=jnp.int32,
                ),
                axis=-1,
            ),
            "real",
            id="take_along_axis",
        ),
        pytest.param(
            lambda v: jax.lax.dynamic_slice_in_dim(v, 1, 2, axis=0),
            "real",
            id="dynamic_slice",
        ),
        pytest.param(_scatter_add, "real", id="scatter_add"),
        pytest.param(
            lambda v: (
                (v + jnp.full_like(v, 0.5))
                .at[jnp.array([0, 1, 2])]
                .set(v[:, :1] + v[:1, :])
            ),
            "real",
            id="scatter_tracked_base_and_updates",
        ),
        pytest.param(_scatter_set, "real", id="scatter_set"),
        pytest.param(_scatter_max, "real", id="scatter_max"),
        pytest.param(_scatter_min, "real", id="scatter_min"),
        pytest.param(
            _scatter_add_duplicate_indices,
            "real",
            id="scatter_add_duplicate",
        ),
        pytest.param(
            _scatter_set_duplicate_indices,
            "real",
            id="scatter_set_duplicate",
        ),
        pytest.param(_scatter_max_base_wins, "real", id="scatter_max_base_wins"),
        pytest.param(_scatter_min_base_wins, "real", id="scatter_min_base_wins"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_indexing(case, op, domain):
    check_unary(op, case, domain=domain)


@parametrize_over_tracked_cases("case")
def test_gather_non_square_feature_axis(case):
    check_unary(
        lambda v: jnp.take(v, jnp.array([1, 0]), axis=2),
        case,
        shape=(3, 3, 2),
    )


@parametrize_over_tracked_cases("case")
def test_scalar_multi_index_gather(case):
    x = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
    check_with_brute_force(
        operator.itemgetter((jnp.array([2, 0]), jnp.array([1, 2]))),
        tracked_case_input(x, case),
    )


@parametrize_over_tracked_cases("case")
def test_scalar_gather_duplicate_indices(case):
    x = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
    check_with_brute_force(
        operator.itemgetter((jnp.array([0, 0, 2]), jnp.array([1, 1, 2]))),
        tracked_case_input(x, case),
    )


@parametrize_over_tracked_cases("case")
def test_dynamic_slice_with_untracked_start(case):
    x = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
    check_with_brute_force(
        lambda value: _dynamic_slice_with_start_arg(value, 1),
        tracked_case_input(x, case),
    )


@parametrize_over_tracked_cases("case")
def test_boolean_mask_gather(case):
    x = random_array(shape=(3, 3), key=20)
    mask = jnp.asarray(x > jnp.mean(x))

    check_with_brute_force(operator.itemgetter(mask), tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_pairwise_upper_triangle_gather(case):
    x = jnp.linspace(-1.0, 1.0, 9, dtype=jnp.float32).reshape(3, 3)

    def upper_triangle_pair_distances(value):
        pair = jnp.sum((value[:, None, :] - value[None, :, :]) ** 2, axis=-1)
        return pair[jnp.triu_indices(value.shape[0], k=1)]

    check_with_brute_force(upper_triangle_pair_distances, tracked_case_input(x, case))
