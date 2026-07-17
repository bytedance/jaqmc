# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Reduction primitive semantics (sum/prod/cumsum/reduce_max/reduce_min)."""

import jax
import jax.numpy as jnp
import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import MATRIX_SHAPE, to_complex, tracked_case_input

from .helpers import check_unary, parametrize_over_tracked_cases


@pytest.mark.parametrize(
    ("op", "domain", "shape"),
    (
        pytest.param(jnp.sum, "real", (3, 3), id="sum"),
        pytest.param(
            lambda v: jnp.sum(v, keepdims=True),
            "real",
            (3, 3),
            id="sum_keepdims",
        ),
        pytest.param(lambda v: jnp.sum(v, axis=0), "real", (3, 3), id="sum_axis0"),
        pytest.param(lambda v: jnp.sum(v, axis=-1), "real", (3, 3), id="sum_last_axis"),
        pytest.param(
            lambda v: jnp.sum(v, axis=(2, 3), keepdims=True),
            "real",
            (3, 3, 2, 4),
            id="sum_keepdims_feature_axes",
        ),
        pytest.param(
            lambda v: jnp.sum(v, axis=0, keepdims=True),
            "real",
            (3, 3),
            id="sum_axis0_keepdims",
        ),
        pytest.param(jnp.prod, "real", (3, 3), id="prod"),
        pytest.param(lambda v: jnp.prod(v, axis=0), "real", (3, 3), id="prod_axis0"),
        pytest.param(lambda v: jnp.cumsum(v, axis=-1), "real", (3, 3), id="cumsum"),
        pytest.param(
            lambda v: jnp.cumsum(v, axis=0), "real", (3, 3), id="cumsum_axis0"
        ),
        pytest.param(
            lambda v: jnp.flip(
                jnp.cumsum(jnp.flip(v, axis=-1), axis=-1),
                axis=-1,
            ),
            "real",
            (3, 3),
            id="cumsum_reverse",
        ),
        pytest.param(
            lambda v: jax.lax.cumsum(v, axis=1, reverse=True),
            "real",
            (3, 3),
            id="lax_cumsum_reverse",
        ),
        pytest.param(
            lambda v: jax.lax.cumsum(v, axis=0, reverse=True),
            "real",
            (3, 3),
            id="lax_cumsum_reverse_axis0",
        ),
        pytest.param(jnp.max, "real", (3, 3), id="reduce_max"),
        pytest.param(
            lambda v: jnp.max(v, axis=0), "real", (3, 3), id="reduce_max_axis0"
        ),
        pytest.param(jnp.min, "real", (3, 3), id="reduce_min"),
        pytest.param(
            lambda v: jnp.min(v, axis=-1), "real", (3, 3), id="reduce_min_last_axis"
        ),
        pytest.param(
            lambda v: jnp.max(v, axis=(0, 1)),
            "real",
            (3, 3),
            id="reduce_max_multi_axis",
        ),
        pytest.param(
            lambda v: jnp.min(v, axis=(0, 1)),
            "real",
            (3, 3),
            id="reduce_min_multi_axis",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_reductions(case, op, domain, shape):
    check_unary(op, case, domain=domain, shape=shape)


@pytest.mark.parametrize(
    ("op", "x"),
    (
        pytest.param(
            lambda value: jnp.max(value, axis=-1),
            jnp.array(
                [
                    [2.0, 2.0, 1.0],
                    [0.0, 3.0, 3.0],
                    [4.0, 1.0, 4.0],
                ],
                dtype=jnp.float32,
            ),
            id="max_ties",
        ),
        pytest.param(
            lambda value: jnp.min(value, axis=-1),
            jnp.array(
                [
                    [-2.0, -2.0, 1.0],
                    [0.0, -3.0, -3.0],
                    [-4.0, 1.0, -4.0],
                ],
                dtype=jnp.float32,
            ),
            id="min_ties",
        ),
        pytest.param(
            lambda value: jnp.max(value, axis=0),
            jnp.array(
                [
                    [2.0, 2.0, 1.0],
                    [0.0, 3.0, 3.0],
                    [4.0, 1.0, 4.0],
                ],
                dtype=jnp.float32,
            ),
            id="max_ties_axis0",
        ),
        pytest.param(
            lambda value: jnp.min(value, axis=0),
            jnp.array(
                [
                    [-2.0, -2.0, 1.0],
                    [0.0, -3.0, -3.0],
                    [-4.0, 1.0, -4.0],
                ],
                dtype=jnp.float32,
            ),
            id="min_ties_axis0",
        ),
        pytest.param(
            lambda value: jnp.max(value, axis=0, keepdims=True),
            jnp.array(
                [
                    [2.0, 2.0, 1.0],
                    [0.0, 3.0, 3.0],
                    [4.0, 1.0, 4.0],
                ],
                dtype=jnp.float32,
            ),
            id="max_ties_axis0_keepdims",
        ),
        pytest.param(
            lambda value: jnp.min(value, axis=-1, keepdims=True),
            jnp.array(
                [
                    [-2.0, -2.0, 1.0],
                    [0.0, -3.0, -3.0],
                    [-4.0, 1.0, -4.0],
                ],
                dtype=jnp.float32,
            ),
            id="min_ties_keepdims",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_reduce_select_ties_split_derivatives(case, op, x):
    check_with_brute_force(op, tracked_case_input(x, case))


@parametrize_over_tracked_cases("case")
def test_prod_with_zeros(case):
    x = jnp.array(
        [
            [1.0, 2.0, 0.5],
            [0.0, 3.0, 1.0],
            [4.0, 0.5, 2.0],
        ],
        dtype=jnp.float32,
    )
    check_with_brute_force(
        lambda value: jnp.prod(value, axis=0),
        tracked_case_input(x, case),
    )


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(jnp.sum, id="complex_sum"),
        pytest.param(jnp.prod, id="complex_prod"),
        pytest.param(jnp.max, id="complex_max"),
        pytest.param(jnp.min, id="complex_min"),
    ),
)
@parametrize_over_tracked_cases("case")
def test_complex_reductions(case, op):
    check_unary(
        lambda packed: op(to_complex(packed) + (1.0 + 0.5j)),
        case,
        shape=MATRIX_SHAPE,
    )
