# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse arithmetic Jacobian behavior."""

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
from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import (
    make_local1_input,
    make_local2_input,
    sparse_local1_input,
    sparse_local2_input,
)

LOGGER_NAME = "jaqmc.laplacian.primitives.core"


def _local1_pair(
    lhs_owner: OwnerRole,
    rhs_owner: OwnerRole,
    *,
    rhs_laplacian: jnp.ndarray | None = None,
) -> tuple[LapTuple, LapTuple]:
    x = jnp.arange(4.0, dtype=jnp.float32).reshape(2, 2)
    lhs = make_local1_input(
        x,
        blocks=jnp.array(
            [[[[1.0, 3.0], [2.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]]],
            dtype=jnp.float32,
        ),
        owners=OwnerRoles(lhs_owner),
        input_shape=(2, 2),
    )
    rhs = make_local1_input(
        x,
        blocks=jnp.array(
            [[[[5.0, 7.0], [6.0, 8.0]], [[0.0, 0.0], [0.0, 0.0]]]],
            dtype=jnp.float32,
        ),
        owners=OwnerRoles(rhs_owner),
        input_shape=(2, 2),
        laplacian=rhs_laplacian,
    )
    return lhs, rhs


def _partial_local1_pair() -> tuple[LapTuple, LapTuple]:
    return _local1_pair(
        OwnerRole(0, np.array([0, 1], dtype=np.int32)),
        OwnerRole(0, np.array([0, 0], dtype=np.int32)),
    )


def _local2_pair(
    *,
    lhs_owners: OwnerRoles,
    rhs_owners: OwnerRoles,
) -> tuple[LapTuple, LapTuple]:
    lhs = make_local2_input(
        jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        blocks=jnp.array(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]],
                [[[4.0, 3.0], [2.0, 1.0]], [[3.5, 2.5], [1.5, 0.5]]],
            ],
            dtype=jnp.float32,
        ),
        owners=lhs_owners,
        input_shape=(2, 2),
        laplacian=jnp.array([[0.25, 0.5], [0.75, 1.0]], dtype=jnp.float32),
    )
    rhs = make_local2_input(
        jnp.array([[4.0, 1.0], [2.0, 3.0]], dtype=jnp.float32),
        blocks=jnp.array(
            [
                [[[2.0, 1.0], [4.0, 3.0]], [[1.5, 0.5], [3.5, 2.5]]],
                [[[5.0, 6.0], [7.0, 8.0]], [[4.5, 5.5], [6.5, 5.5]]],
            ],
            dtype=jnp.float32,
        ),
        owners=rhs_owners,
        input_shape=(2, 2),
        laplacian=jnp.array([[1.0, 0.75], [0.5, 0.25]], dtype=jnp.float32),
    )
    return lhs, rhs


def _permuted_local2_pair() -> tuple[LapTuple, LapTuple]:
    """Build Local2 inputs with the same roles stored in opposite slots."""
    owner_a = OwnerRole(0, np.array([0, 1], dtype=np.int32))
    owner_b = OwnerRole(1, np.array([0, 1], dtype=np.int32))
    return _local2_pair(
        lhs_owners=OwnerRoles(owner_a, owner_b),
        rhs_owners=OwnerRoles(owner_b, owner_a),
    )


def _incompatible_local2_pair() -> tuple[LapTuple, LapTuple]:
    """Build Local2 inputs whose union requires three distinct owner roles."""
    owner_a = OwnerRole(0, np.array([0, 1], dtype=np.int32))
    owner_b = OwnerRole(1, np.array([0, 1], dtype=np.int32))
    owner_c = OwnerRole(0, np.array([1, 0], dtype=np.int32))
    return _local2_pair(
        lhs_owners=OwnerRoles(owner_a, owner_b),
        rhs_owners=OwnerRoles(owner_c, owner_b),
    )


def _overlapping_local2_pair() -> tuple[LapTuple, LapTuple]:
    """Build Local2 inputs whose duplicate A support overlaps an A/B support."""
    owner_a = OwnerRole(0, np.array([0, 1], dtype=np.int32))
    owner_b = OwnerRole(1, np.array([0, 1], dtype=np.int32))
    return _local2_pair(
        lhs_owners=OwnerRoles(owner_a, owner_a),
        rhs_owners=OwnerRoles(owner_a, owner_b),
    )


def _repeated_owner_local2_seed() -> LapTuple:
    x = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    return make_local2_input(
        x,
        blocks=jnp.array([[[[-1.0], [2.0]]], [[[3.0], [5.0]]]], dtype=jnp.float32),
        owners=OwnerRoles(
            OwnerRole(None, np.array([0], dtype=np.int32)),
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
        ),
        input_shape=(2, 1),
    )


_LOCAL1_PAIR_SHAPE = ((2, 2), (2, 2))


@pytest.mark.parametrize(
    ("op", "lhs_owner", "rhs_owner", "make_args", "expected_jacobian", "fallback"),
    (
        pytest.param(
            operator.add,
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(0, np.array([1, 0], dtype=np.int32)),
            None,
            Local2Jacobian,
            None,
            id="mismatched_local1_add",
        ),
        pytest.param(
            operator.sub,
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(0, np.array([1, 0], dtype=np.int32)),
            None,
            Local2Jacobian,
            None,
            id="mismatched_local1_sub",
        ),
        pytest.param(
            operator.mul,
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(0, np.array([1, 0], dtype=np.int32)),
            None,
            Local2Jacobian,
            None,
            id="mismatched_local1_mul",
        ),
        pytest.param(
            operator.mul,
            None,
            None,
            _partial_local1_pair,
            Local2Jacobian,
            None,
            id="partially_matching_local1_mul",
        ),
        pytest.param(
            operator.add,
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(1, np.array([0, 1], dtype=np.int32)),
            None,
            Local2Jacobian,
            None,
            id="cross_axis_local1_add",
        ),
        pytest.param(
            operator.sub,
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(1, np.array([0, 1], dtype=np.int32)),
            None,
            Local2Jacobian,
            None,
            id="cross_axis_local1_sub",
        ),
        pytest.param(
            operator.mul,
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(1, np.array([0, 1], dtype=np.int32)),
            None,
            Local2Jacobian,
            None,
            id="cross_axis_local1_mul",
        ),
        pytest.param(
            operator.add,
            None,
            None,
            _incompatible_local2_pair,
            None,
            (logging.DEBUG, "unrepresentable"),
            id="unrepresentable_local2_add",
        ),
        pytest.param(
            operator.sub,
            None,
            None,
            _incompatible_local2_pair,
            None,
            (logging.DEBUG, "unrepresentable"),
            id="unrepresentable_local2_sub",
        ),
        pytest.param(
            operator.mul,
            None,
            None,
            _incompatible_local2_pair,
            None,
            (logging.DEBUG, "unrepresentable"),
            id="unrepresentable_local2_mul",
        ),
        pytest.param(
            operator.add,
            None,
            None,
            _permuted_local2_pair,
            Local2Jacobian,
            None,
            id="permuted_local2_add",
        ),
        pytest.param(
            operator.sub,
            None,
            None,
            _permuted_local2_pair,
            Local2Jacobian,
            None,
            id="permuted_local2_sub",
        ),
        pytest.param(
            operator.mul,
            None,
            None,
            _permuted_local2_pair,
            Local2Jacobian,
            None,
            id="permuted_local2_mul",
        ),
    ),
)
def test_sparse_binary_jacobian(
    op,
    lhs_owner,
    rhs_owner,
    make_args,
    expected_jacobian,
    fallback,
    caplog,
):
    if make_args is None:
        output_shape, input_shape = _LOCAL1_PAIR_SHAPE
        args = (
            sparse_local1_input(
                lhs_owner, output_shape=output_shape, input_shape=input_shape
            ),
            sparse_local1_input(
                rhs_owner, output_shape=output_shape, input_shape=input_shape
            ),
        )
    else:
        args = make_args()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(op)(*args)
    if expected_jacobian is None:
        assert isinstance(actual.jacobian, jnp.ndarray)
    else:
        assert isinstance(actual.jacobian, expected_jacobian)
    check_with_brute_force(op, *args, actual_result=actual)
    if fallback is not None:
        [record] = [
            record
            for record in caplog.records
            if "dense-fallback[" in record.getMessage()
        ]
        assert record.levelno == fallback[0]
        assert fallback[1] in record.getMessage()


@pytest.mark.parametrize("op", (operator.add, operator.sub, operator.mul))
def test_overlapping_local2_supports_stay_sparse_under_jit(op):
    """Duplicate support slots coalesce into a representable Local2Jacobian."""
    lhs, rhs = _overlapping_local2_pair()
    transformed = forward_laplacian(op)
    blocks = jax.jit(lambda: transformed(lhs, rhs).jacobian.blocks)()
    actual = transformed(lhs, rhs)
    assert isinstance(actual.jacobian, Local2Jacobian)
    np.testing.assert_allclose(blocks, actual.jacobian.blocks)
    check_with_brute_force(op, lhs, rhs, actual_result=actual)


@pytest.mark.parametrize(
    ("op", "make_args", "expected_jacobian"),
    (
        pytest.param(
            operator.add,
            lambda: (
                (
                    make_laplacian_input(
                        jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                        sparse_axis=0,
                    ),
                )
                * 2
            ),
            Local1Jacobian,
            id="matching_local1_add",
        ),
        pytest.param(
            operator.sub,
            lambda: (
                (
                    make_laplacian_input(
                        jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                        sparse_axis=0,
                    ),
                )
                * 2
            ),
            Local1Jacobian,
            id="matching_local1_sub",
        ),
        pytest.param(
            operator.mul,
            lambda: (
                (
                    make_laplacian_input(
                        jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                        sparse_axis=0,
                    ),
                )
                * 2
            ),
            Local1Jacobian,
            id="matching_local1_mul",
        ),
        pytest.param(
            operator.add,
            lambda: (
                sparse_local2_input(),
                sparse_local2_input(),
            ),
            Local2Jacobian,
            id="matching_local2_add",
        ),
        pytest.param(
            operator.sub,
            lambda: (
                sparse_local2_input(),
                sparse_local2_input(),
            ),
            Local2Jacobian,
            id="matching_local2_sub",
        ),
        pytest.param(
            operator.mul,
            lambda: (
                sparse_local2_input(),
                sparse_local2_input(),
            ),
            Local2Jacobian,
            id="matching_local2_mul",
        ),
    ),
)
def test_matching_sparse_binary_operations(op, make_args, expected_jacobian, caplog):
    args = make_args()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(op)(*args)
    assert isinstance(actual.jacobian, expected_jacobian)
    check_with_brute_force(op, *args, actual_result=actual)


@pytest.mark.parametrize(
    ("fn", "make_args", "expected_jacobian"),
    (
        pytest.param(
            lambda value: value + 1.0,
            sparse_local2_input,
            Local2Jacobian,
            id="local2_add_plain",
        ),
        pytest.param(
            lambda value: value * value,
            _repeated_owner_local2_seed,
            Local2Jacobian,
            id="repeated_owner_local2_mul",
        ),
        pytest.param(
            lambda value: value**2.5,
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3) + 1.0,
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_pow_plain",
        ),
        pytest.param(
            lambda value: value / 2.0,
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3) + 1.0,
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_divide_plain_rhs",
        ),
        pytest.param(
            lambda value: 3.0 / value,
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3) + 1.0,
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_divide_sparse_rhs",
        ),
        pytest.param(
            lambda value: value % 1.5,
            lambda: make_laplacian_input(
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3) + 1.0,
                sparse_axis=0,
            ),
            Local1Jacobian,
            id="local1_remainder_plain",
        ),
    ),
)
def test_sparse_unary_arithmetic(fn, make_args, expected_jacobian, caplog):
    seed = make_args()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, expected_jacobian)
    check_with_brute_force(fn, seed, actual_result=actual)


def test_mixed_local1_local2_multiply_falls_back(caplog):
    local1 = sparse_local1_input(
        OwnerRole(0, np.array([0, 1], dtype=np.int32)),
        output_shape=(2, 2),
        input_shape=(2, 2),
    )
    _, local2 = _incompatible_local2_pair()
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(operator.mul)(local1, local2)
    assert isinstance(actual.jacobian, jnp.ndarray)
    check_with_brute_force(operator.mul, local1, local2, actual_result=actual)
    [record] = [
        record for record in caplog.records if "dense-fallback[" in record.getMessage()
    ]
    assert record.levelno == logging.WARNING
    assert "not_implemented" in record.getMessage()


def test_local2_add_rejects_incompatible_tracked_input_metadata():
    """Binary sparse rules cannot silently combine different derivative bases."""
    lhs, _ = _permuted_local2_pair()
    rhs = make_local2_input(
        lhs.x,
        blocks=jnp.ones((2, 3, *lhs.x.shape), dtype=jnp.float32),
        owners=lhs.jacobian.owners,
        input_shape=(2, 3),
    )

    with pytest.raises(
        ValueError,
        match="Incompatible Local2Jacobian tracked-input metadata for add/sub",
    ):
        forward_laplacian(operator.add)(lhs, rhs)


@pytest.mark.parametrize(
    ("lhs_owner", "rhs_owner", "expected_jacobian"),
    (
        pytest.param(1, 1, Local1Jacobian, id="matching_constant_owners"),
        pytest.param(0, 2, Local2Jacobian, id="distinct_constant_owners"),
    ),
)
def test_constant_owner_multiply(lhs_owner, rhs_owner, expected_jacobian, caplog):
    lhs = sparse_local1_input(OwnerRole(None, np.array([lhs_owner], dtype=np.int32)))
    rhs = sparse_local1_input(OwnerRole(None, np.array([rhs_owner], dtype=np.int32)))
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    actual = forward_laplacian(operator.mul)(lhs, rhs)
    assert isinstance(actual.jacobian, expected_jacobian)
    check_with_brute_force(operator.mul, lhs, rhs, actual_result=actual)
