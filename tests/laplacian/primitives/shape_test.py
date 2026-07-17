# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Shape primitive semantics (reshape/transpose/slice/concat/rev/split).

Ops are written shape-generically over the trailing axes because the lifted
input rank differs per case (``local2`` inputs carry an extra owner axis).
"""

import operator

import jax
import jax.numpy as jnp
import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import MATRIX_SHAPE, to_complex, tracked_case_input

from .helpers import check_unary, parametrize_over_tracked_cases


def _split_recombine(v):
    a, b, c = jnp.split(v, 3, axis=-1)
    return a + b - 2.0 * c


def _split_unequal(v):
    a, b = jnp.split(v, [1], axis=-1)
    return jnp.sum(jnp.sin(a)) + jnp.sum(jnp.cos(b))


@pytest.mark.parametrize(
    ("op", "domain", "shape"),
    (
        pytest.param(
            lambda v: v.reshape(v.shape[0], -1),
            "real",
            (3, 3, 4),
            id="reshape_merge_trailing_rank3",
        ),
        pytest.param(
            lambda v: v.reshape(*v.shape[:-2], -1),
            "real",
            (3, 3),
            id="reshape_merge_last_two",
        ),
        pytest.param(
            lambda v: v.reshape(v.shape[0], v.shape[1], -1),
            "real",
            (3, 3, 4),
            id="reshape_preserve_leading_axes",
        ),
        pytest.param(lambda v: v.reshape(-1), "real", (3, 3), id="reshape_flatten"),
        pytest.param(
            lambda v: v.reshape(*v.shape[:-1], 3, 1),
            "real",
            (3, 3),
            id="reshape_split_last",
        ),
        pytest.param(
            lambda v: jax.lax.reshape(v, (v.shape[1], v.shape[0]), dimensions=(1, 0)),
            "real",
            (3, 3),
            id="reshape_permute",
        ),
        pytest.param(lambda v: jnp.swapaxes(v, -1, -2), "real", (3, 3), id="transpose"),
        pytest.param(
            lambda v: jnp.swapaxes(v, 0, 2),
            "real",
            (3, 3, 4),
            id="transpose_rank3_nonadjacent",
        ),
        pytest.param(
            lambda v: jnp.moveaxis(v, 2, 0),
            "real",
            (3, 3, 2),
            id="transpose_non_square_feature_to_front",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_reshape_and_transpose(case, op, domain, shape):
    check_unary(op, case, domain=domain, shape=shape)


@pytest.mark.parametrize(
    ("op", "domain", "shape"),
    (
        pytest.param(
            operator.itemgetter(slice(1, None)), "real", (3, 3), id="slice_axis0"
        ),
        pytest.param(
            operator.itemgetter(slice(1, 3, 2)),
            "real",
            (3, 3),
            id="slice_axis0_stride",
        ),
        pytest.param(
            operator.itemgetter((slice(None, None, 2), slice(None, None, 2))),
            "real",
            (3, 3),
            id="slice_leading_axes_stride",
        ),
        pytest.param(
            operator.itemgetter((..., slice(2))),
            "real",
            (3, 3),
            id="slice_last_axis",
        ),
        pytest.param(
            operator.itemgetter((..., slice(1, 5, 2))),
            "real",
            (4, 4, 6),
            id="slice_feature_stride",
        ),
        pytest.param(
            lambda v: jnp.squeeze(v[..., None], axis=-1),
            "real",
            (3, 3),
            id="squeeze",
        ),
        pytest.param(
            lambda v: jnp.squeeze(v, axis=2),
            "real",
            (3, 3, 1),
            id="squeeze_existing_singleton",
        ),
        pytest.param(
            lambda v: jnp.squeeze(v, axis=(0, 1)),
            "real",
            (1, 1, 4),
            id="squeeze_leading_singletons",
        ),
        pytest.param(
            lambda v: jnp.flip(v, axis=1),
            "real",
            (3, 3, 4),
            id="rev_middle_axis_rank3",
        ),
        pytest.param(
            lambda v: jnp.flip(v, axis=-1),
            "real",
            (3, 3),
            id="rev_last_axis",
        ),
        pytest.param(lambda v: jnp.flip(v, axis=0), "real", (3, 3), id="rev_axis0"),
        pytest.param(_split_recombine, "real", (3, 3), id="split_recombine"),
        pytest.param(
            lambda v: jnp.split(v, 3, axis=-1)[1] ** 2,
            "real",
            (3, 3),
            id="split_take_part",
        ),
        pytest.param(
            lambda v: jnp.split(v, [1, 2], axis=0)[1],
            "real",
            (3, 3),
            id="split_axis0",
        ),
        pytest.param(_split_unequal, "real", (3, 3), id="split_unequal"),
        pytest.param(
            lambda v: jnp.split(v, 2, axis=-2)[0],
            "real",
            (4, 4, 6),
            id="split_negative_axis",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_slice_reverse_squeeze_and_split(case, op, domain, shape):
    check_unary(op, case, domain=domain, shape=shape)


@pytest.mark.parametrize(
    ("op", "domain", "shape"),
    (
        pytest.param(
            lambda v: jnp.broadcast_to(v, (3, 3, v.shape[-1])),
            "real",
            (1, 1, 4),
            id="broadcast_leading_singletons",
        ),
        pytest.param(
            lambda v: jnp.broadcast_to(
                v[..., None, :], (*v.shape[:-1], 2, v.shape[-1])
            ),
            "real",
            (3, 3),
            id="broadcast_in_dim",
        ),
        pytest.param(
            lambda v: jnp.broadcast_to(v[:, None, :], (v.shape[0], 2, v.shape[1])),
            "real",
            (3, 3),
            id="broadcast_between_owner_axes",
        ),
        pytest.param(
            lambda v: jnp.broadcast_to(v[None, ...], (2, *v.shape)),
            "real",
            (3, 3),
            id="broadcast_leading",
        ),
        pytest.param(
            lambda v: jnp.concatenate([v, 2.0 * v], axis=-1),
            "real",
            (3, 3),
            id="concat_last_axis",
        ),
        pytest.param(
            lambda v: jnp.concatenate([v, v], axis=0),
            "real",
            (3, 3),
            id="concat_axis0",
        ),
        pytest.param(
            lambda v: jnp.concatenate([v, 2.0 * v], axis=1),
            "real",
            (3, 3),
            id="concat_axis1",
        ),
        pytest.param(
            lambda v: jnp.concatenate([v, v], axis=-2),
            "real",
            (3, 3),
            id="concat_negative_axis",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_broadcast_and_concatenate(case, op, domain, shape):
    check_unary(op, case, domain=domain, shape=shape)


@parametrize_over_tracked_cases("case")
def test_concat_feature_axis_with_plain_segment(case):
    x = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3)

    def prepend_plain_feature_planes(value):
        return jnp.concatenate(
            [
                jnp.zeros((value.shape[0], value.shape[1], 2), dtype=value.dtype),
                value,
            ],
            axis=2,
        )

    check_with_brute_force(prepend_plain_feature_planes, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_concat_plain_feature_columns(case):
    x = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3)

    def append_constant_features(value):
        return jnp.concatenate(
            [
                value,
                jnp.zeros((*value.shape[:2], 2), dtype=value.dtype),
            ],
            axis=2,
        )

    check_with_brute_force(append_constant_features, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_concat_three_segments_with_plain_middle(case):
    x = jnp.arange(27.0, dtype=jnp.float32).reshape(3, 3, 3)

    def concat_with_plain_middle(value):
        return jnp.concatenate(
            [
                value,
                jnp.zeros((value.shape[0], value.shape[1], 2), dtype=value.dtype),
                2.0 * value,
            ],
            axis=2,
        )

    check_with_brute_force(concat_with_plain_middle, tracked_case_input(x, case))


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(lambda m: m.reshape(*m.shape[:-2], -1), id="complex_reshape"),
        pytest.param(lambda m: jnp.swapaxes(m, -1, -2), id="complex_transpose"),
        pytest.param(
            operator.itemgetter((..., slice(1, None), slice(None))),
            id="complex_slice",
        ),
        pytest.param(
            lambda m: jnp.concatenate([m, 2.0 * m], axis=-1),
            id="complex_concat",
        ),
        pytest.param(lambda m: jnp.split(m, 2, axis=-1)[0] ** 2, id="complex_split"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_complex_shape_ops(case, op):
    check_unary(
        lambda packed: op(to_complex(packed) + (1.0 + 0.5j)),
        case,
        shape=MATRIX_SHAPE,
    )
