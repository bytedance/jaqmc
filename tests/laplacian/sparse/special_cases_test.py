# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse topology edge cases whose Forward Laplacian results match brute force."""

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
    make_laplacian_input,
)
from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.sparse.helpers import (
    broadcast_local1_seed,
    mismatched_local1_pair,
    repeated_owner_ids_local1_seed,
    repeated_owner_local2_seed,
    select_n_broadcast_mixed_branches_scenario,
    two_local1_query_key_dot_scenario,
)


class TestBroadcastedLocal1Layout:
    """Broadcast-filled Local1 blocks with owner role on axis 1."""

    def test_reduce_sum_matches_brute_force(self):
        check_with_brute_force(
            lambda value: jnp.sum(value, axis=0),
            broadcast_local1_seed(),
        )

    def test_reduce_max_matches_brute_force(self):
        check_with_brute_force(
            lambda value: jnp.max(value, axis=0),
            broadcast_local1_seed(),
        )

    def test_slice_leading_broadcast_axis_matches_brute_force(self):
        check_with_brute_force(
            operator.itemgetter(slice(1, None)),
            broadcast_local1_seed(),
        )

    def test_concatenate_leading_broadcast_axis_matches_brute_force(self):
        seed = broadcast_local1_seed()

        def concat_doubled(value):
            return jnp.concatenate([value, value], axis=0)

        check_with_brute_force(concat_doubled, seed)


class TestSparseBroadcastThenTransform:
    """Identity-seeded Local1 followed by broadcast and a shape op."""

    @staticmethod
    def _identity_seed() -> LapTuple:
        return make_laplacian_input(
            jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
            sparse_axis=0,
        )

    def test_reduce_sum_matches_brute_force(self):
        seed = self._identity_seed()
        check_with_brute_force(
            lambda value: jnp.sum(
                jnp.broadcast_to(value, (4, *value.shape)),
                axis=0,
            ),
            seed,
        )

    def test_slice_matches_brute_force(self):
        seed = self._identity_seed()
        check_with_brute_force(
            lambda value: jnp.broadcast_to(value, (4, *value.shape))[:, 1:],
            seed,
        )

    def test_concatenate_matches_brute_force(self):
        seed = self._identity_seed()

        def broadcast_then_concatenate(value):
            broadcast = jnp.broadcast_to(value, (2, *value.shape))
            return jnp.concatenate([broadcast, broadcast], axis=0)

        check_with_brute_force(broadcast_then_concatenate, seed)

    def test_reshape_matches_brute_force(self):
        seed = self._identity_seed()
        check_with_brute_force(
            lambda value: jnp.reshape(
                jnp.broadcast_to(value, (2, *value.shape)),
                (2, 6),
            ),
            seed,
        )

    def test_dot_general_matches_brute_force(self):
        seed = self._identity_seed()

        def broadcast_then_dot(value):
            broadcast = jnp.broadcast_to(value, (4, *value.shape))
            return jax.lax.dot_general(
                broadcast,
                jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5),
                dimension_numbers=(((2,), (0,)), ((), ())),
            )

        check_with_brute_force(broadcast_then_dot, seed)


class TestRepeatedOwnerIdsLocal1:
    """Local1 state whose owner role repeats ids along an output axis."""

    def test_reduce_sum_matches_brute_force(self):
        check_with_brute_force(
            lambda value: jnp.sum(value, axis=0),
            repeated_owner_ids_local1_seed(),
        )

    def test_reduce_max_matches_brute_force(self):
        check_with_brute_force(
            lambda value: jnp.max(value, axis=0),
            repeated_owner_ids_local1_seed(),
        )

    def test_dot_matches_brute_force(self):
        check_with_brute_force(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
                dimension_numbers=(((0,), (0,)), ((), ())),
            ),
            repeated_owner_ids_local1_seed(),
        )


class TestOwnerRemappingGathers:
    def test_reordered_gather_matches_brute_force(self):
        x = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
        blocks = jnp.arange(18.0, dtype=jnp.float32).reshape(1, 2, 3, 3)
        seed = LapTuple(
            x,
            Local1Jacobian(
                blocks=blocks,
                owners=OwnerRoles(OwnerRole(0, np.array([2, 0, 1], dtype=np.int32))),
                input_shape=(3, 2),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        check_with_brute_force(
            operator.itemgetter(jnp.array([2, 0, 1], dtype=jnp.int32)),
            seed,
        )

    def test_higher_rank_indices_factorize_each_local2_role(self):
        x = jnp.arange(18.0, dtype=jnp.float32).reshape(3, 3, 2)
        seed = LapTuple(
            x,
            Local2Jacobian(
                blocks=jnp.arange(108.0, dtype=jnp.float32).reshape(2, 3, 3, 3, 2),
                owners=OwnerRoles(
                    OwnerRole(0, np.arange(3, dtype=np.int32)),
                    OwnerRole(1, np.arange(3, dtype=np.int32)),
                ),
                input_shape=(3, 3),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        start_indices = jnp.array(
            [
                [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]]],
                [[[0, 1, 0], [1, 1, 0]], [[0, 1, 0], [1, 1, 0]]],
            ],
            dtype=jnp.int32,
        )

        check_with_brute_force(
            lambda value: jax.lax.gather(
                value,
                start_indices,
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(),
                    collapsed_slice_dims=(0, 1, 2),
                    start_index_map=(0, 1, 2),
                ),
                slice_sizes=(1, 1, 1),
            ),
            seed,
        )

    def test_constant_owner_gather_matches_brute_force(self):
        x = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
        seed = LapTuple(
            x,
            Local1Jacobian(
                blocks=jnp.arange(24.0, dtype=jnp.float32).reshape(1, 2, 3, 4),
                owners=OwnerRoles(OwnerRole(None, np.array([1], dtype=np.int32))),
                input_shape=(3, 2),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        check_with_brute_force(
            operator.itemgetter(jnp.array([2, 0, 2], dtype=jnp.int32)),
            seed,
        )


class TestMismatchedLocal1BinaryCorrectness:
    """Mismatched Local1 owner roles match the dense oracle for arithmetic."""

    @pytest.mark.parametrize(
        "op",
        (
            pytest.param(operator.add, id="add"),
            pytest.param(operator.sub, id="sub"),
            pytest.param(operator.mul, id="multiply"),
        ),
    )
    def test_matches_brute_force(self, op):
        local1_lhs, local1_rhs = mismatched_local1_pair()
        check_with_brute_force(op, local1_lhs, local1_rhs)


class TestPartialOwnerMatchLocal1:
    """Binary Local1 operands where one owner role only partially matches.

    The left operand maps output rows to owners ``[0, 1]`` while the right maps
    both rows to owner ``0`` via ``[0, 0]``. This is neither a fully matching
    Local1 pair nor the fully mismatched swapped-owner case above, so it
    protects the promotion path for mixed shared/distinct ownership.
    """

    def test_mul_matches_brute_force(self):
        x = jnp.arange(4.0, dtype=jnp.float32).reshape(2, 2)
        local1_lhs = LapTuple(
            x,
            Local1Jacobian(
                blocks=jnp.array(
                    [
                        [
                            [[1.0, 3.0], [2.0, 4.0]],
                            [[0.0, 0.0], [0.0, 0.0]],
                        ]
                    ],
                    dtype=jnp.float32,
                ),
                owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
                input_shape=(2, 2),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        local1_rhs = LapTuple(
            x,
            Local1Jacobian(
                blocks=jnp.array(
                    [
                        [
                            [[5.0, 7.0], [6.0, 8.0]],
                            [[0.0, 0.0], [0.0, 0.0]],
                        ]
                    ],
                    dtype=jnp.float32,
                ),
                owners=OwnerRoles(OwnerRole(0, np.array([0, 0], dtype=np.int32))),
                input_shape=(2, 2),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        check_with_brute_force(operator.mul, local1_lhs, local1_rhs)


class TestRemappedLocal2Gathers:
    def test_scalar_multi_index_gather_matches_brute_force(self):
        x = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3)
        seed = LapTuple(
            x,
            Local2Jacobian(
                blocks=jnp.arange(288.0, dtype=jnp.float32).reshape(2, 3, 4, 4, 3),
                owners=OwnerRoles(
                    OwnerRole(0, np.array([3, 1, 0, 2], dtype=np.int32)),
                    OwnerRole(1, np.array([2, 0, 3, 1], dtype=np.int32)),
                ),
                input_shape=(4, 3),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        check_with_brute_force(
            operator.itemgetter(
                (jnp.array([3, 1, 3]), jnp.array([2, 0, 2]), jnp.array([1, 2, 1]))
            ),
            seed,
        )


class TestRepeatedOwnerLocal2:
    def test_mul_matches_brute_force(self):
        check_with_brute_force(
            lambda value: value * value,
            repeated_owner_local2_seed(),
        )


class TestSelectNBroadcastMixedBranches:
    """Production Local1 topology with a zero-local branch adaptation.

    This is not just the generic select_n primitive matrix: it uses the sparse
    topology produced by ``make_laplacian_input(..., sparse_axis=0)`` together
    with mixed tracked/plain branches whose local structure must be adapted.
    """

    def test_matches_brute_force(self):
        fn, seed = select_n_broadcast_mixed_branches_scenario()
        check_with_brute_force(fn, seed)


class TestTwoLocal1DotCorrectness:
    def test_query_key_dot_matches_brute_force(self):
        fn, seed = two_local1_query_key_dot_scenario()
        check_with_brute_force(fn, seed)


class TestMixedRepresentationConcatenate:
    """Concatenate operands that mix sparse families or dense/sparse layouts."""

    @staticmethod
    def _shared_operand_pair() -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3)
        return x[:2], x[2:]

    def test_dense_and_local1_matches_brute_force(self):
        lhs_x, rhs_x = self._shared_operand_pair()
        dense_lhs = make_laplacian_input(lhs_x)
        local1_rhs = make_laplacian_input(rhs_x, sparse_axis=0)

        def concat_dense_local1(left, right):
            return jnp.concatenate([left, right], axis=0)

        check_with_brute_force(concat_dense_local1, dense_lhs, local1_rhs)

    def test_local1_and_local2_matches_brute_force(self):
        lhs_x, rhs_x = self._shared_operand_pair()
        local1_lhs = make_laplacian_input(lhs_x, sparse_axis=0)
        local2_rhs = LapTuple(
            rhs_x,
            Local2Jacobian(
                blocks=jnp.arange(36.0, dtype=jnp.float32).reshape(2, 3, 2, 3),
                owners=OwnerRoles(
                    OwnerRole(0, np.arange(2, dtype=np.int32)),
                    OwnerRole(0, np.arange(2, dtype=np.int32)),
                ),
                input_shape=(2, 3),
                input_owner_axis=0,
            ),
            jnp.zeros_like(rhs_x),
        )

        def concat_local1_local2(left, right):
            return jnp.concatenate([left, right], axis=0)

        check_with_brute_force(concat_local1_local2, local1_lhs, local2_rhs)
