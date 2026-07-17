# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse indexing Jacobian behavior."""

import logging
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    LapTuple,
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    forward_laplacian,
    make_laplacian_input,
)
from tests.laplacian.helpers import check_sparse_jacobian, check_with_brute_force
from tests.laplacian.input_fixtures import (
    make_local1_input,
    make_local2_input,
    sparse_local1_input,
    sparse_local2_input,
)

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


@pytest.mark.parametrize(
    ("fn", "make_seed", "expected_jacobian"),
    (
        pytest.param(
            operator.itemgetter(jnp.array([1, 0], dtype=jnp.int32)),
            lambda: sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32))),
            Local1Jacobian,
            id="constant_owner_reorder",
        ),
        pytest.param(
            operator.itemgetter(jnp.array([1, 0], dtype=jnp.int32)),
            lambda: sparse_local1_input(
                OwnerRole(0, np.array([2, 2], dtype=np.int32)),
                output_shape=(2, 2),
                input_shape=(3, 2),
            ),
            Local1Jacobian,
            id="repeated_owner_reorder",
        ),
        pytest.param(
            lambda value: jax.lax.gather(
                value,
                jnp.array([[5]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,),
                ),
                slice_sizes=(1, value.shape[1]),
                mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
                fill_value=-7.0,
            ),
            lambda: sparse_local1_input(
                OwnerRole(0, np.array([2, 2], dtype=np.int32)),
                output_shape=(2, 2),
                input_shape=(3, 2),
            ),
            Local1Jacobian,
            id="fill_out_of_bounds",
        ),
        pytest.param(
            operator.itemgetter(
                (jnp.array([3, 1, 3]), jnp.array([2, 0, 2]), jnp.array([1, 2, 1]))
            ),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_scalar_multi_index",
        ),
        pytest.param(
            lambda value: jax.lax.gather(
                value,
                jnp.array([[3, 2], [1, 0]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(0,),
                    collapsed_slice_dims=(0, 2),
                    start_index_map=(0, 2),
                ),
                slice_sizes=(1, value.shape[1], 1),
            ),
            sparse_local2_input,
            Local2Jacobian,
            id="local2_interleaved_offsets",
        ),
    ),
)
def test_sparse_gather_jacobian(fn, make_seed, expected_jacobian):
    seed = make_seed()
    check_sparse_jacobian(fn, seed, expected_jacobian=expected_jacobian)


def test_reordered_owner_role_gather():
    x = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
    seed = make_local1_input(
        x,
        blocks=jnp.arange(18.0, dtype=jnp.float32).reshape(1, 2, *x.shape),
        owners=OwnerRoles(OwnerRole(0, np.array([2, 0, 1], dtype=np.int32))),
        input_shape=(3, 2),
    )
    fn = operator.itemgetter(jnp.array([2, 0, 1], dtype=jnp.int32))
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, Local1Jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)


def test_higher_rank_indices_factorize_local2_owner_roles():
    x = jnp.arange(18.0, dtype=jnp.float32).reshape(3, 3, 2)
    seed = make_local2_input(
        x,
        blocks=jnp.arange(108.0, dtype=jnp.float32).reshape(2, 3, *x.shape),
        owners=OwnerRoles(
            OwnerRole(0, np.arange(3, dtype=np.int32)),
            OwnerRole(1, np.arange(3, dtype=np.int32)),
        ),
        input_shape=(3, 3),
    )
    start_indices = jnp.array(
        [
            [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]]],
            [[[0, 1, 0], [1, 1, 0]], [[0, 1, 0], [1, 1, 0]]],
        ],
        dtype=jnp.int32,
    )
    fn = lambda value: jax.lax.gather(
        value,
        start_indices,
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0, 1, 2),
            start_index_map=(0, 1, 2),
        ),
        slice_sizes=(1, 1, 1),
    )
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, Local2Jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)


def test_traced_owner_axis_gather():
    seed = sparse_local1_input(OwnerRole(None, np.array([1], dtype=np.int32)))
    indices = jnp.array([1, 0], dtype=jnp.int32)
    transformed = forward_laplacian(lambda value, index: value[index])
    actual = LapTuple(
        *jax.jit(
            lambda index: (
                transformed(seed, index).x,
                transformed(seed, index).dense_jacobian,
                transformed(seed, index).laplacian,
            )
        )(indices)
    )
    check_with_brute_force(
        operator.itemgetter(indices),
        seed,
        actual_result=actual,
    )


def test_traced_non_owner_gather_retains_local1():
    def gather_features(value: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        return value[:, indices]

    x = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
    seed = make_local1_input(
        x,
        blocks=jnp.arange(24.0, dtype=jnp.float32).reshape(1, 2, *x.shape),
        owners=OwnerRoles(OwnerRole(0, np.arange(3, dtype=np.int32))),
        input_shape=(3, 2),
        laplacian=jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4) / 10.0,
    )
    indices = jnp.array([3, 1], dtype=jnp.int32)
    transformed = forward_laplacian(gather_features)
    actual = LapTuple(
        *jax.jit(
            lambda index: (
                transformed(seed, index).x,
                transformed(seed, index).dense_jacobian,
                transformed(seed, index).laplacian,
            )
        )(indices)
    )
    check_with_brute_force(
        lambda value: gather_features(value, indices),
        seed,
        actual_result=actual,
    )


def test_traced_collapsed_owner_gather_retains_local1():
    """A traced gather may collapse a fixed owner axis without densifying."""
    x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
    seed = make_local1_input(
        x,
        blocks=jnp.ones((1, 1, *x.shape), dtype=jnp.float32),
        owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
        input_shape=(2, 1),
    )

    def gather_feature(value, index):
        return jax.lax.gather(
            value,
            index[None],
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0, 1),
                start_index_map=(1,),
            ),
            slice_sizes=(1, 1),
        )

    transformed = forward_laplacian(gather_feature)
    blocks = jax.jit(lambda index: transformed(seed, index).jacobian.blocks)(
        jnp.array(1, dtype=jnp.int32)
    )
    actual = LapTuple(
        *jax.jit(
            lambda index: (
                transformed(seed, index).x,
                transformed(seed, index).dense_jacobian,
                transformed(seed, index).laplacian,
            )
        )(jnp.array(1, dtype=jnp.int32))
    )

    assert blocks.shape == (1, 1)
    check_with_brute_force(
        lambda value: gather_feature(value, jnp.array(1, dtype=jnp.int32)),
        seed,
        actual_result=actual,
    )


def test_traced_batched_owner_gather_falls_back_to_dense():
    """A gather batching the owner axis cannot retain one-axis owner metadata."""
    x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
    seed = make_local1_input(
        x,
        blocks=jnp.ones((1, 1, *x.shape), dtype=jnp.float32),
        owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
        input_shape=(2, 1),
    )

    def gather_batched_features(value, indices):
        return jax.lax.gather(
            value,
            indices[:, None],
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(1,),
                start_index_map=(1,),
                operand_batching_dims=(0,),
                start_indices_batching_dims=(0,),
            ),
            slice_sizes=(1, 1),
        )

    indices = jnp.array([0, 2], dtype=jnp.int32)
    transformed = forward_laplacian(gather_batched_features)
    jacobian = jax.jit(lambda index: transformed(seed, index).jacobian)(indices)
    actual = LapTuple(
        *jax.jit(
            lambda index: (
                transformed(seed, index).x,
                transformed(seed, index).dense_jacobian,
                transformed(seed, index).laplacian,
            )
        )(indices)
    )

    assert isinstance(jacobian, jnp.ndarray)
    assert actual.dense_jacobian.shape == (2, 2)
    check_with_brute_force(
        lambda value: gather_batched_features(value, indices),
        seed,
        actual_result=actual,
    )


def test_one_hot_gather_reports_missing_jvp_rule():
    seed = make_laplacian_input(
        jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2),
        sparse_axis=0,
    )

    def gather_rows(value):
        return jax.lax.gather(
            value,
            jnp.array([[0], [2]], dtype=jnp.int32),
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,),
            ),
            slice_sizes=(1, value.shape[1]),
            mode=jax.lax.GatherScatterMode.ONE_HOT,
        )

    with pytest.raises(NotImplementedError, match="does not provide a JVP rule"):
        forward_laplacian(gather_rows)(seed)


@pytest.mark.parametrize(
    ("fn", "make_seed", "indices", "expected_shape"),
    (
        pytest.param(
            operator.itemgetter(
                (
                    jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
                    jnp.zeros((2, 2), dtype=jnp.int32),
                )
            ),
            lambda: make_laplacian_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                sparse_axis=0,
            ),
            None,
            (9, 2, 2),
            id="checkerboard_owner",
        ),
        pytest.param(
            lambda value: jax.lax.gather(
                value,
                jnp.array([[0], [1]], dtype=jnp.int32),
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, 2),
                    collapsed_slice_dims=(),
                    start_index_map=(0,),
                ),
                slice_sizes=(2, value.shape[1]),
            ),
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            None,
            (12, 2, 2, 3),
            id="moving_owner_window",
        ),
        pytest.param(
            operator.itemgetter(
                (
                    jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
                    jnp.array([[0, 0], [1, 1]], dtype=jnp.int32),
                )
            ),
            lambda: make_local2_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                blocks=jnp.ones((2, 1, 3, 3), dtype=jnp.float32),
                owners=OwnerRoles(
                    OwnerRole(0, np.arange(3, dtype=np.int32)),
                    OwnerRole(1, np.arange(3, dtype=np.int32)),
                ),
                input_shape=(3, 1),
            ),
            None,
            (3, 2, 2),
            id="one_local2_role",
        ),
        pytest.param(
            lambda value, index: value[index],
            lambda: make_laplacian_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                sparse_axis=0,
            ),
            jnp.array([2, 0], dtype=jnp.int32),
            (9, 2, 3),
            id="owner_selecting",
        ),
        pytest.param(
            lambda value, start_indices: jax.lax.gather(
                value,
                start_indices[:, None],
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(1,),
                    start_index_map=(1,),
                ),
                slice_sizes=(0, 1),
            ),
            lambda: forward_laplacian(operator.itemgetter(slice(0)))(
                make_laplacian_input(
                    jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2),
                    sparse_axis=0,
                )
            ),
            jnp.array([0], dtype=jnp.int32),
            (6, 1, 0),
            id="empty_owner_role",
        ),
    ),
)
def test_unrepresentable_gathers_fall_back(
    fn,
    make_seed,
    indices,
    expected_shape,
    caplog,
):
    seed = make_seed()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    transformed = forward_laplacian(fn)
    out = (
        transformed(seed)
        if indices is None
        else jax.jit(lambda index: transformed(seed, index))(indices)
    )
    assert isinstance(out.jacobian, jnp.ndarray)
    assert out.dense_jacobian.shape == expected_shape
    [record] = [
        record
        for record in caplog.records
        if "dense-fallback[gather]" in record.getMessage()
    ]
    assert record.levelno == logging.DEBUG
    assert "unrepresentable" in record.getMessage()
