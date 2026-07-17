# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Known sparse-preserving operations keep a structured sparse Jacobian.

Retention is intentionally type-only here. Some sparse-preserving graphs keep
the same family, while others transition to a richer representable family such
as ``Local1Jacobian`` to ``Local2Jacobian``; numerical correctness for the same
graphs belongs in primitive or sparse special-case tests.
"""

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
from tests.laplacian.sparse.helpers import (
    assert_retains_sparse_family,
    broadcast_local1_seed,
    mismatched_local1_pair,
    repeated_owner_ids_local1_seed,
    repeated_owner_local2_seed,
    select_n_broadcast_mixed_branches_scenario,
    two_local1_query_key_dot_scenario,
)


def _make_local2_pair_seed() -> LapTuple:
    """Return a pairwise Local2 state used by multiple retention groups."""
    x = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3)
    return LapTuple(
        x,
        Local2Jacobian(
            blocks=jnp.arange(288.0, dtype=jnp.float32).reshape(2, 3, 4, 4, 3),
            owners=OwnerRoles(
                OwnerRole(0, np.arange(4, dtype=np.int32)),
                OwnerRole(1, np.arange(4, dtype=np.int32)),
            ),
            input_shape=(4, 3),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )


def _make_complex_local1_seed() -> LapTuple:
    """Return a complex Local1 state for unary representation checks."""
    real = jnp.linspace(-1.0, 1.0, 12, dtype=jnp.float32).reshape(4, 3)
    imag = jnp.linspace(0.5, -0.5, 12, dtype=jnp.float32).reshape(4, 3)
    return make_laplacian_input(real + 1j * imag, sparse_axis=0)


def _make_complex_local2_pair_seed() -> LapTuple:
    """Return a complex Local2 state for unary representation checks."""
    real = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3)
    imag = jnp.arange(48.0, 96.0, dtype=jnp.float32).reshape(4, 4, 3) / 10.0
    blocks_real = jnp.arange(288.0, dtype=jnp.float32).reshape(2, 3, 4, 4, 3)
    blocks_imag = (
        jnp.arange(288.0, 576.0, dtype=jnp.float32).reshape(2, 3, 4, 4, 3) / 10.0
    )
    x = real + 1j * imag
    return LapTuple(
        x,
        Local2Jacobian(
            blocks=blocks_real + 1j * blocks_imag,
            owners=OwnerRoles(
                OwnerRole(0, np.arange(4, dtype=np.int32)),
                OwnerRole(1, np.arange(4, dtype=np.int32)),
            ),
            input_shape=(4, 3),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )


class TestShapeAndIndexingRetention:
    @pytest.mark.parametrize(
        ("fn", "x"),
        (
            pytest.param(
                lambda value: jnp.transpose(value, (1, 0, 2)),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="transpose",
            ),
            pytest.param(
                lambda value: jnp.reshape(value, (4, 1, 4, 3)),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="reshape",
            ),
            pytest.param(
                lambda value: jnp.broadcast_to(value, (2, *value.shape)),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="broadcast",
            ),
            pytest.param(
                lambda value: jnp.squeeze(value, axis=2),
                jnp.arange(16.0, dtype=jnp.float32).reshape(4, 4, 1),
                id="squeeze",
            ),
            pytest.param(
                lambda value: jnp.flip(value, axis=0),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="reverse_owner_axis",
            ),
            pytest.param(
                operator.itemgetter(slice(None, None, 2)),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="strided_slice_owner_axis",
            ),
            pytest.param(
                lambda value: jnp.concatenate([value, 2.0 * value], axis=2),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="concatenate_off_owner_axis",
            ),
            pytest.param(
                lambda value: jnp.concatenate([value[1:], value[:1]], axis=0),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="concatenate_owner_axis",
            ),
            pytest.param(
                operator.itemgetter(
                    (jnp.array([3, 1]), jnp.array([2, 0]), jnp.array([1, 2]))
                ),
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                id="scalar_multi_index_gather",
            ),
            pytest.param(
                operator.itemgetter(jnp.array([3, 1])),
                jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
                id="row_gather",
            ),
            pytest.param(
                lambda value: jnp.take(value, jnp.array([1, 0]), axis=2),
                jnp.arange(24.0, dtype=jnp.float32).reshape(4, 3, 2),
                id="feature_gather",
            ),
            pytest.param(
                lambda value: jax.lax.gather(
                    value,
                    jnp.array(
                        [[[0], [2], [1]], [[3], [1], [0]]],
                        dtype=jnp.int32,
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
                jnp.arange(8.0, dtype=jnp.float32).reshape(2, 4),
                id="batched_non_owner_gather",
            ),
        ),
    )
    def test_local1_operations_retain_family(self, fn, x):
        seed = make_laplacian_input(x, sparse_axis=0)
        out = forward_laplacian(fn)(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    @pytest.mark.parametrize(
        "fn",
        (
            pytest.param(
                lambda value: jnp.transpose(value, (1, 0, 2)),
                id="transpose",
            ),
            pytest.param(
                lambda value: jnp.reshape(value, (4, 4, 1, 3)),
                id="reshape",
            ),
            pytest.param(
                lambda value: jnp.broadcast_to(value, (2, *value.shape)),
                id="broadcast",
            ),
            pytest.param(
                lambda value: jnp.flip(value, axis=0),
                id="reverse_owner_axis",
            ),
            pytest.param(
                operator.itemgetter((slice(None), slice(None), slice(1, None))),
                id="slice_feature_axis",
            ),
            pytest.param(
                lambda value: jnp.concatenate([value[:, :2], value[:, 2:]], axis=1),
                id="concatenate_owner_axis",
            ),
            pytest.param(
                lambda value: jnp.concatenate(
                    [jnp.zeros((4, 4, 2), dtype=value.dtype), value],
                    axis=2,
                ),
                id="concatenate_plain_segment",
            ),
            pytest.param(
                operator.itemgetter(
                    (jnp.array([3, 1]), jnp.array([2, 0]), jnp.array([1, 2]))
                ),
                id="scalar_multi_index_gather",
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
                id="interleaved_gather_offsets",
            ),
        ),
    )
    def test_local2_operations_retain_family(self, fn):
        out = forward_laplacian(fn)(_make_local2_pair_seed())
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_squeeze_retains_local2_family(self):
        x = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 1, 4, 3)
        seed = LapTuple(
            x,
            Local2Jacobian(
                blocks=jnp.arange(288.0, dtype=jnp.float32).reshape(2, 3, 4, 1, 4, 3),
                owners=OwnerRoles(
                    OwnerRole(0, np.arange(4, dtype=np.int32)),
                    OwnerRole(2, np.arange(4, dtype=np.int32)),
                ),
                input_shape=(4, 3),
                input_owner_axis=0,
            ),
            jnp.zeros_like(x),
        )
        out = forward_laplacian(lambda value: jnp.squeeze(value, axis=1))(seed)
        assert_retains_sparse_family(out, Local2Jacobian)

    @pytest.mark.parametrize(
        ("seed", "axis", "expected_family"),
        (
            pytest.param(
                make_laplacian_input(
                    jnp.linspace(-1.0, 1.0, 12).reshape(4, 3),
                    sparse_axis=0,
                ),
                0,
                Local1Jacobian,
                id="local1_owner_axis",
            ),
            pytest.param(
                make_laplacian_input(
                    jnp.linspace(-1.0, 1.0, 12).reshape(4, 3),
                    sparse_axis=0,
                ),
                1,
                Local1Jacobian,
                id="local1_non_owner_axis",
            ),
            pytest.param(
                _make_local2_pair_seed(),
                0,
                Local2Jacobian,
                id="local2_owner_axis",
            ),
            pytest.param(
                _make_local2_pair_seed(),
                2,
                Local2Jacobian,
                id="local2_feature_axis",
            ),
        ),
    )
    def test_split_retains_family(self, seed, axis, expected_family):
        if getattr(jax.lax, "split_p", None) is None:
            pytest.skip("jax.lax.split_p is unavailable")

        first, second = forward_laplacian(
            lambda value: tuple(jnp.split(value, [1], axis=axis))
        )(seed)
        assert_retains_sparse_family(first, expected_family)
        assert_retains_sparse_family(second, expected_family)


class TestBroadcastedLocal1Retention:
    def test_reduce_sum_retains_local1_family(self):
        out = forward_laplacian(lambda value: jnp.sum(value, axis=0))(
            broadcast_local1_seed()
        )
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_slice_leading_broadcast_axis_retains_local1_family(self):
        out = forward_laplacian(operator.itemgetter(slice(1, None)))(
            broadcast_local1_seed()
        )
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_concatenate_leading_broadcast_axis_retains_local1_family(self):
        seed = broadcast_local1_seed()
        out = forward_laplacian(lambda value: jnp.concatenate([value, value], axis=0))(
            seed
        )
        assert_retains_sparse_family(out, Local1Jacobian)


class TestUnaryRetention:
    @pytest.mark.parametrize(
        ("fn", "seed", "expected_family"),
        (
            pytest.param(
                jnp.exp,
                make_laplacian_input(
                    jnp.linspace(-1.0, 1.0, 12).reshape(4, 3),
                    sparse_axis=0,
                ),
                Local1Jacobian,
                id="elementwise_local1",
            ),
            pytest.param(
                jnp.exp,
                _make_local2_pair_seed(),
                Local2Jacobian,
                id="elementwise_local2",
            ),
            pytest.param(
                operator.neg,
                _make_local2_pair_seed(),
                Local2Jacobian,
                id="neg_local2",
            ),
            pytest.param(
                lambda value: jax.lax.convert_element_type(value, jnp.float32),
                make_laplacian_input(
                    jnp.arange(12.0, dtype=jnp.float16).reshape(4, 3),
                    sparse_axis=0,
                ),
                Local1Jacobian,
                id="float_cast_local1",
            ),
            pytest.param(
                lambda value: jax.lax.convert_element_type(value, jnp.complex64),
                _make_local2_pair_seed(),
                Local2Jacobian,
                id="complex_cast_local2",
            ),
            pytest.param(
                jnp.conj,
                _make_complex_local1_seed(),
                Local1Jacobian,
                id="conj_local1",
            ),
            pytest.param(
                jnp.conj,
                _make_complex_local2_pair_seed(),
                Local2Jacobian,
                id="conj_local2",
            ),
            pytest.param(
                jnp.real,
                _make_complex_local1_seed(),
                Local1Jacobian,
                id="real_local1",
            ),
            pytest.param(
                jnp.real,
                _make_complex_local2_pair_seed(),
                Local2Jacobian,
                id="real_local2",
            ),
            pytest.param(
                jnp.imag,
                _make_complex_local1_seed(),
                Local1Jacobian,
                id="imag_local1",
            ),
            pytest.param(
                jnp.imag,
                _make_complex_local2_pair_seed(),
                Local2Jacobian,
                id="imag_local2",
            ),
        ),
    )
    def test_operations_retain_family(self, fn, seed, expected_family):
        out = forward_laplacian(fn)(seed)
        assert_retains_sparse_family(out, expected_family)

    def test_round_retains_local1_family(self):
        out = forward_laplacian(
            lambda value: jax.lax.round(
                value,
                rounding_method=jax.lax.RoundingMethod.AWAY_FROM_ZERO,
            )
        )(broadcast_local1_seed())
        assert_retains_sparse_family(out, Local1Jacobian)


class TestArithmeticRetention:
    @pytest.mark.parametrize(
        "op",
        (
            pytest.param(operator.add, id="add"),
            pytest.param(operator.sub, id="sub"),
            pytest.param(operator.mul, id="multiply"),
        ),
    )
    def test_matching_local1_binary_operations_retain_family(self, op):
        seed = make_laplacian_input(
            jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
            sparse_axis=0,
        )
        out = forward_laplacian(op)(seed, seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    @pytest.mark.parametrize(
        "op",
        (
            pytest.param(operator.add, id="add"),
            pytest.param(operator.sub, id="sub"),
            pytest.param(operator.mul, id="multiply"),
        ),
    )
    def test_matching_local2_binary_operations_retain_family(self, op):
        seed = _make_local2_pair_seed()
        out = forward_laplacian(op)(seed, seed)
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_sparse_plain_add_retains_local2_family(self):
        out = forward_laplacian(lambda value: value + 1.0)(_make_local2_pair_seed())
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_repeated_owner_local2_mul_retains_local2_family(self):
        out = forward_laplacian(lambda value: value * value)(
            repeated_owner_local2_seed()
        )
        assert_retains_sparse_family(out, Local2Jacobian)

    @pytest.mark.parametrize(
        "fn",
        (
            pytest.param(lambda value: value**2.5, id="pow"),
            pytest.param(lambda value: value / 2.0, id="divide_plain_rhs"),
            pytest.param(lambda value: 3.0 / value, id="divide_sparse_rhs"),
            pytest.param(lambda value: value % 1.5, id="remainder"),
        ),
    )
    def test_sparse_plain_operations_retain_local1_family(self, fn):
        seed = make_laplacian_input(
            jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3) + 1.0,
            sparse_axis=0,
        )
        out = forward_laplacian(fn)(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    @pytest.mark.parametrize(
        "op",
        (
            pytest.param(operator.add, id="add"),
            pytest.param(operator.sub, id="sub"),
            pytest.param(operator.mul, id="multiply"),
        ),
    )
    def test_mismatched_local1_binary_operations_promote_to_local2(self, op):
        local1_lhs, local1_rhs = mismatched_local1_pair()
        out = forward_laplacian(op)(local1_lhs, local1_rhs)
        assert_retains_sparse_family(out, Local2Jacobian)


class TestComplexRetention:
    def test_matching_local1_complex_retains_family(self):
        seed = make_laplacian_input(
            jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
            sparse_axis=0,
        )
        out = forward_laplacian(jax.lax.complex)(seed, seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_matching_local2_complex_retains_family(self):
        seed = _make_local2_pair_seed()
        out = forward_laplacian(jax.lax.complex)(seed, seed)
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_mismatched_local1_complex_promotes_to_local2(self):
        local1_real, local1_imag = mismatched_local1_pair()
        out = forward_laplacian(jax.lax.complex)(local1_real, local1_imag)
        assert_retains_sparse_family(out, Local2Jacobian)


class TestReductionRetention:
    def test_reduce_sum_non_owner_axis_retains_local1_family(self):
        seed = make_laplacian_input(
            jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
            sparse_axis=0,
        )
        out = forward_laplacian(lambda value: jnp.sum(value, axis=1))(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_reduce_sum_feature_axis_retains_local2_family(self):
        out = forward_laplacian(lambda value: jnp.sum(value, axis=2))(
            _make_local2_pair_seed()
        )
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_repeated_owner_ids_reduce_sum_retains_local1_family(self):
        out = forward_laplacian(lambda value: jnp.sum(value, axis=0))(
            repeated_owner_ids_local1_seed()
        )
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_repeated_owner_ids_reduce_max_retains_local1_family(self):
        out = forward_laplacian(lambda value: jnp.max(value, axis=0))(
            repeated_owner_ids_local1_seed()
        )
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_reduce_max_feature_axis_retains_local2_family(self):
        out = forward_laplacian(lambda value: jnp.max(value, axis=2))(
            _make_local2_pair_seed()
        )
        assert_retains_sparse_family(out, Local2Jacobian)


class TestDotGeneralRetention:
    def test_sparse_left_plain_dot_general_retains_local1_family(self):
        seed = make_laplacian_input(
            jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4),
            sparse_axis=1,
        )
        out = forward_laplacian(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(10.0, dtype=jnp.float32).reshape(2, 5) / 10.0,
                dimension_numbers=(((0,), (0,)), ((), ())),
            )
        )(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_sparse_right_plain_dot_general_retains_local1_family(self):
        seed = make_laplacian_input(
            jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4),
            sparse_axis=1,
        )
        out = forward_laplacian(
            lambda value: jax.lax.dot_general(
                jnp.arange(10.0, dtype=jnp.float32).reshape(5, 2) / 10.0,
                value,
                dimension_numbers=(((1,), (0,)), ((), ())),
            )
        )(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_local2_plain_dot_general_retains_local2_family(self):
        out = forward_laplacian(
            lambda value: (
                value @ (jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2) / 10.0)
            )
        )(_make_local2_pair_seed())
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_batched_sparse_plain_dot_general_retains_local1_family(self):
        seed = make_laplacian_input(
            jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4),
            sparse_axis=0,
        )
        out = forward_laplacian(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(40.0, dtype=jnp.float32).reshape(2, 4, 5),
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            )
        )(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_broadcast_sparse_plain_dot_general_retains_local1_family(self):
        seed = make_laplacian_input(
            jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
            sparse_axis=0,
        )
        out = forward_laplacian(
            lambda value: jax.lax.dot_general(
                jnp.broadcast_to(value, (4, *value.shape)),
                jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5),
                dimension_numbers=(((2,), (0,)), ((), ())),
            )
        )(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_batched_local1_dot_general_retains_local1_family(self):
        lhs = make_laplacian_input(
            jnp.arange(18.0, dtype=jnp.float32).reshape(2, 3, 3),
            sparse_axis=0,
        )
        rhs = make_laplacian_input(
            jnp.arange(18.0, dtype=jnp.float32).reshape(2, 3, 3)[::-1],
            sparse_axis=0,
        )
        out = forward_laplacian(
            lambda left, right: jax.lax.dot_general(
                left,
                right,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            )
        )(lhs, rhs)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_batched_local2_dot_general_retains_local2_family(self):
        seed = _make_local2_pair_seed()
        out = forward_laplacian(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3),
                dimension_numbers=(((2,), (2,)), ((0, 1), (0, 1))),
            )
        )(seed)
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_repeated_owner_ids_dot_retains_local1_family(self):
        out = forward_laplacian(
            lambda value: jax.lax.dot_general(
                value,
                jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
                dimension_numbers=(((0,), (0,)), ((), ())),
            )
        )(repeated_owner_ids_local1_seed())
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_two_local1_dot_promotes_to_local2(self):
        fn, seed = two_local1_query_key_dot_scenario()
        out = forward_laplacian(fn)(seed)
        assert_retains_sparse_family(out, Local2Jacobian)


class TestSelectionRetention:
    def test_sparse_plain_maximum_retains_local1_family(self):
        seed = make_laplacian_input(
            jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
            sparse_axis=0,
        )
        out = forward_laplacian(lambda value: jnp.maximum(value, 1.5))(seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_matching_local1_maximum_retains_family(self):
        seed = make_laplacian_input(
            jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3),
            sparse_axis=0,
        )
        out = forward_laplacian(jnp.maximum)(seed, seed)
        assert_retains_sparse_family(out, Local1Jacobian)

    def test_select_n_with_plain_branch_retains_local2_family(self):
        out = forward_laplacian(
            lambda value: jax.lax.select_n(
                jnp.array(0, dtype=jnp.int32),
                value,
                jnp.zeros((4, 4, 3), dtype=value.dtype),
            )
        )(_make_local2_pair_seed())
        assert_retains_sparse_family(out, Local2Jacobian)

    def test_select_n_broadcast_mixed_branches_retains_local1_family(self):
        fn, seed = select_n_broadcast_mixed_branches_scenario()
        out = forward_laplacian(fn)(seed)
        assert_retains_sparse_family(out, Local1Jacobian)
