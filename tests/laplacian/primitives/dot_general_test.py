# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""dot_general primitive semantics (matmul/dot/vdot/einsum)."""

import jax
import jax.numpy as jnp
import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import (
    MATRIX_SHAPE,
    random_array,
    to_complex,
    tracked_case_input,
)

from .helpers import (
    check_binary,
    check_unary,
    parametrize_over_binary_cases,
    parametrize_over_tracked_cases,
)


def _polynomial(v):
    flat = v.reshape(-1)
    w1 = random_array("unit", (flat.size, 4))
    w2 = random_array("unit", (flat.size, flat.size, 4))
    return jnp.sum(flat @ w1 + jnp.einsum("i,j,ijk->k", flat, flat, w2))


@pytest.mark.parametrize(
    ("op", "domain", "shape"),
    (
        pytest.param(
            lambda v: jnp.tanh(v @ random_array("unit", (3, 2))),
            "real",
            (3, 3),
            id="matvec_tanh",
        ),
        pytest.param(
            lambda v: (random_array("unit", (2, 3)) + 0.5) @ v,
            "real",
            (3, 3),
            id="matmul_left_const",
        ),
        pytest.param(
            lambda v: v @ random_array("unit", (3,)),
            "real",
            (3, 3),
            id="dot_last_axis",
        ),
        pytest.param(
            lambda v: jnp.vdot(v, random_array("unit", v.shape) + 0.4),
            "real",
            (3, 3),
            id="vdot",
        ),
        pytest.param(
            lambda v: jnp.einsum("...i,i->...", v, random_array("unit", (3,))),
            "real",
            (3, 3),
            id="einsum_contract_last",
        ),
        pytest.param(
            lambda v: jnp.einsum("...ij,ij->...", v, random_array("unit", (3, 3))),
            "real",
            (3, 3),
            id="einsum_contract_matrix",
        ),
        pytest.param(
            lambda v: jnp.einsum("...i,j->...ij", v, random_array("unit", (4,), key=5)),
            "real",
            (3, 3),
            id="einsum_outer_const_rhs",
        ),
        pytest.param(
            lambda v: jnp.sum(jnp.einsum("...i,...j->...ij", v, v)),
            "real",
            (3, 3),
            id="einsum_outer_quadratic",
        ),
        pytest.param(_polynomial, "real", (3, 3), id="polynomial"),
        pytest.param(
            lambda v: jax.lax.dot_general(
                v,
                random_array("unit", (4, 5)),
                dimension_numbers=(((2,), (0,)), ((), ())),
            ),
            "real",
            (3, 3, 4),
            id="contract_feature_axis",
        ),
        pytest.param(
            lambda v: jax.lax.dot_general(
                v,
                random_array("unit", (2, 5)),
                dimension_numbers=(((2,), (0,)), ((), ())),
            ),
            "real",
            (3, 3, 2),
            id="contract_non_square_feature_axis",
        ),
        pytest.param(
            lambda v: jax.lax.dot_general(
                v,
                random_array("unit", (3, 5)),
                dimension_numbers=(((0,), (0,)), ((), ())),
            ),
            "real",
            (3, 3, 2),
            id="contract_leading_axis",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_closed_over_weight_contractions(case, op, domain, shape):
    check_unary(op, case, domain=domain, shape=shape)


@pytest.mark.parametrize(
    ("op", "domain"),
    (
        pytest.param(
            lambda a, b: jnp.sum(jnp.tanh(a @ jnp.swapaxes(b, -1, -2))),
            "real",
            id="matmul_tracked_tracked",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.einsum("...i,...i->...", a, b) ** 2),
            "real",
            id="contract_last_tracked",
        ),
        pytest.param(
            lambda a, b: jax.lax.dot_general(
                a,
                b,
                dimension_numbers=(((0, 1), (0, 1)), ((), ())),
            ),
            "real",
            id="full_contract_scalar",
        ),
        pytest.param(
            lambda a, b: jnp.sum(jnp.einsum("...i,...j->...ij", a, b)),
            "real",
            id="outer_product_tracked",
        ),
    ),
)
@parametrize_over_binary_cases(("lhs", "rhs"))
def test_tracked_contractions(lhs, rhs, op, domain):
    check_binary(op, lhs, rhs, domain=domain)


@pytest.mark.parametrize(
    ("op", "lhs_shape", "rhs_shape"),
    (
        pytest.param(
            lambda a, b: jax.lax.dot_general(
                a,
                b,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            ),
            (4, 4, 4),
            (4, 4, 5),
            id="batched_matmul",
        ),
        pytest.param(
            lambda a, b: jax.lax.dot_general(
                a,
                b,
                dimension_numbers=(((2,), (2,)), ((0, 1), (0, 1))),
            ),
            (4, 4, 2),
            (4, 4, 2, 5),
            id="batched_non_square_feature_contract",
        ),
        pytest.param(
            lambda a, b: jax.lax.dot_general(
                a,
                b,
                dimension_numbers=(((2, 3), (0, 1)), ((), ())),
            ),
            (4, 4, 4, 4),
            (4, 4, 6),
            id="multi_contract",
        ),
        pytest.param(
            lambda a, b: jax.lax.dot_general(
                a,
                b,
                dimension_numbers=(((2, 3), (1, 2)), ((0,), (0,))),
            ),
            (4, 4, 4, 4),
            (4, 4, 4, 4),
            id="batched_multi_contract",
        ),
        pytest.param(
            lambda a, b: jax.lax.dot_general(
                a,
                b,
                dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
            ),
            (4, 4, 4, 5),
            (4, 4, 5, 5),
            id="two_batch_axes_contract",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_two_tracked_dimension_number_variants(case, op, lhs_shape, rhs_shape):
    lhs = random_array("real", lhs_shape)
    rhs = random_array("real", rhs_shape, key=1)
    check_with_brute_force(
        op,
        tracked_case_input(lhs, case, key=2, input_shape=(4, 5)),
        tracked_case_input(rhs, case, key=3, input_shape=(4, 5)),
        rtol=1e-4,
        atol=1e-4,
    )


@parametrize_over_tracked_cases("case")
def test_batched_matvec_left_tracked_right_plain(case):
    def op(value):
        return jax.lax.dot_general(
            value,
            random_array("unit", (4, 4), key=6),
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        )

    check_unary(op, case, shape=(4, 4, 4))


@parametrize_over_tracked_cases("case")
def test_batched_matvec_left_plain_right_tracked(case):
    def op(rhs):
        return jax.lax.dot_general(
            random_array("unit", (4, 4), key=7),
            rhs,
            dimension_numbers=(((1,), (2,)), ((0,), (0,))),
        )

    check_unary(op, case, shape=(4, 4, 4))


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(
            lambda m: jnp.sum(jnp.tanh(random_array("complex", (2, 2)) @ m)),
            id="complex_matmul_const",
        ),
        pytest.param(
            lambda m: jnp.sum(m @ random_array("complex", 2)),
            id="complex_dot_const",
        ),
        pytest.param(
            lambda m: jnp.vdot(m, random_array("complex", m.shape)),
            id="complex_vdot_flat",
        ),
        pytest.param(
            lambda m: jnp.einsum("...ij,ij->...", m, random_array("complex", (2, 2))),
            id="complex_einsum_contract",
        ),
    ),
)
@parametrize_over_tracked_cases("case")
def test_complex_constant_contractions(case, op):
    check_unary(
        lambda packed: op(to_complex(packed) + (1.0 + 0.5j)),
        case,
        shape=MATRIX_SHAPE,
    )


@parametrize_over_binary_cases(("lhs", "rhs"))
def test_complex_tracked_contractions(lhs, rhs):
    check_binary(
        lambda pa, pb: jnp.sum(
            jnp.tanh(to_complex(pa) @ jnp.swapaxes(to_complex(pb), -1, -2))
        ),
        lhs,
        rhs,
        shape=MATRIX_SHAPE,
    )


@parametrize_over_binary_cases(("lhs", "rhs"))
def test_complex_tracked_vdot(lhs, rhs):
    check_binary(
        lambda pa, pb: jnp.vdot(
            to_complex(pa) + (1.0 + 0.5j),
            to_complex(pb) + (0.5 + 0.25j),
        ),
        lhs,
        rhs,
        shape=MATRIX_SHAPE,
    )
