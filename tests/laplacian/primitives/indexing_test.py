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


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(
            lambda v: jax.lax.gather(
                v,
                jnp.array([[2, 3], [0, 1]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0, 2),
                    start_index_map=(0, 2),
                ),
                slice_sizes=(1, v.shape[1], 1),
            ),
            id="offset_collapsed",
        ),
        pytest.param(
            lambda v: jax.lax.gather(
                v,
                jnp.array([[2, 3], [0, 1]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(0,),
                    collapsed_slice_dims=(0, 2),
                    start_index_map=(0, 2),
                ),
                slice_sizes=(1, v.shape[1], 1),
            ),
            id="interleaved_offsets",
        ),
        pytest.param(
            lambda v: jax.lax.gather(
                v,
                jnp.array([[1, 1, 0]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(0, 1),
                    collapsed_slice_dims=(2,),
                    start_index_map=(0, 1, 2),
                ),
                slice_sizes=(2, 2, 1),
            ),
            id="partial_windows",
        ),
        pytest.param(
            lambda v: jax.lax.gather(
                v,
                jnp.array([[-3], [2]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, 2),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,),
                ),
                slice_sizes=(1, v.shape[1], v.shape[2]),
                mode="clip",
            ),
            id="clip",
        ),
        pytest.param(
            lambda v: jax.lax.gather(
                v,
                jnp.array([[0], [4]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, 2),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,),
                ),
                slice_sizes=(1, v.shape[1], v.shape[2]),
                mode="fill",
                fill_value=13.0,
            ),
            id="fill",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_general_gather_layouts_match_brute_force(case, op):
    x = jnp.arange(36.0, dtype=jnp.float32).reshape(3, 3, 4)
    check_with_brute_force(op, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_gather_batching_dims_matches_brute_force(case):
    x = jnp.arange(16.0, dtype=jnp.float32).reshape(4, 4)
    check_with_brute_force(
        lambda v: jax.lax.gather(
            v,
            jnp.broadcast_to(
                jnp.array([[0], [2], [1]], dtype=jnp.int32),
                (4, 3, 1),
            ),
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(1,),
                start_index_map=(1,),
                operand_batching_dims=(0,),
                start_indices_batching_dims=(0,),
            ),
            slice_sizes=(1, 1),
        ),
        tracked_case_input(x, case, input_shape=(4, 3)),
    )
