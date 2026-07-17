# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse complex primitive Jacobian behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaqmc.laplacian.hessian as laplacian_hessian
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
    make_local2_input,
    sparse_local1_input,
    sparse_local2_input,
)


def _complex_local2_seed() -> LapTuple:
    real = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 4, 3)
    imag = jnp.arange(48.0, 96.0, dtype=jnp.float32).reshape(4, 4, 3) / 10.0
    x = real + 1j * imag
    return make_local2_input(
        x,
        blocks=(
            jnp.arange(288.0, dtype=jnp.float32).reshape(2, 3, *x.shape)
            + 1j
            * jnp.arange(288.0, 576.0, dtype=jnp.float32).reshape(2, 3, *x.shape)
            / 10.0
        ),
        owners=OwnerRoles(
            OwnerRole(0, np.arange(4, dtype=np.int32)),
            OwnerRole(1, np.arange(4, dtype=np.int32)),
        ),
        input_shape=(4, 3),
    )


def _real_local2_seed() -> LapTuple:
    return sparse_local2_input()


@pytest.mark.parametrize(
    ("fn", "make_seed", "expected_jacobian", "projection"),
    (
        pytest.param(
            jnp.conj,
            lambda: make_laplacian_input(
                jnp.linspace(-1.0, 1.0, 12, dtype=jnp.float32).reshape(4, 3)
                + 1j * jnp.linspace(0.5, -0.5, 12, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            None,
            id="local1_conj",
        ),
        pytest.param(
            jnp.conj,
            _complex_local2_seed,
            Local2Jacobian,
            None,
            id="local2_conj",
        ),
        pytest.param(
            jnp.real,
            lambda: make_laplacian_input(
                jnp.linspace(-1.0, 1.0, 12, dtype=jnp.float32).reshape(4, 3)
                + 1j * jnp.linspace(0.5, -0.5, 12, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            jnp.real,
            id="local1_real",
        ),
        pytest.param(
            jnp.real,
            _complex_local2_seed,
            Local2Jacobian,
            jnp.real,
            id="local2_real",
        ),
        pytest.param(
            jnp.imag,
            lambda: make_laplacian_input(
                jnp.linspace(-1.0, 1.0, 12, dtype=jnp.float32).reshape(4, 3)
                + 1j * jnp.linspace(0.5, -0.5, 12, dtype=jnp.float32).reshape(4, 3),
                sparse_axis=0,
            ),
            Local1Jacobian,
            jnp.imag,
            id="local1_imag",
        ),
        pytest.param(
            jnp.imag,
            _complex_local2_seed,
            Local2Jacobian,
            jnp.imag,
            id="local2_imag",
        ),
    ),
)
def test_sparse_complex_unary_operations(
    fn,
    make_seed,
    expected_jacobian,
    projection,
):
    seed = make_seed()
    actual = forward_laplacian(fn)(seed)
    assert isinstance(actual.jacobian, expected_jacobian)
    if projection is None:
        check_with_brute_force(fn, seed, actual_result=actual)
        return
    # The generic oracle rejects real outputs from complex primals because its
    # complex Hessian path is holomorphic-only. These projections are linear in
    # the packed complex derivative payload, so assert their observable Forward
    # Laplacian contract directly instead.
    np.testing.assert_allclose(actual.x, projection(seed.x))
    np.testing.assert_allclose(
        actual.dense_jacobian,
        projection(seed.dense_jacobian),
    )
    np.testing.assert_allclose(actual.laplacian, projection(seed.laplacian))


@pytest.mark.parametrize(
    ("make_args", "expected_jacobian"),
    (
        pytest.param(
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
            id="matching_local1",
        ),
        pytest.param(
            lambda: (_real_local2_seed(), _real_local2_seed()),
            Local2Jacobian,
            id="matching_local2",
        ),
        pytest.param(
            lambda: (
                sparse_local1_input(
                    OwnerRole(0, np.array([0, 1], dtype=np.int32)),
                    output_shape=(2, 2),
                    input_shape=(2, 2),
                ),
                sparse_local1_input(
                    OwnerRole(1, np.array([0, 1], dtype=np.int32)),
                    output_shape=(2, 2),
                    input_shape=(2, 2),
                ),
            ),
            Local2Jacobian,
            id="cross_axis_local1",
        ),
    ),
)
def test_sparse_complex_binary_operations(make_args, expected_jacobian):
    real, imag = make_args()
    actual = forward_laplacian(jax.lax.complex)(real, imag)
    assert isinstance(actual.jacobian, expected_jacobian)
    check_with_brute_force(jax.lax.complex, real, imag, actual_result=actual)


def test_linear_complex_avoids_generic_hessian(monkeypatch):
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    seed = LapTuple(x, jnp.ones((3, *x.shape)), jnp.zeros_like(x))

    def fail_if_hessian_is_materialized(*args, **kwargs):
        raise AssertionError("linear complex primitive materialized a Hessian")

    monkeypatch.setattr(
        laplacian_hessian,
        "JHJ_via_hessian",
        fail_if_hessian_is_materialized,
    )
    result = forward_laplacian(
        lambda real: jax.lax.complex(real, jnp.zeros_like(real))
    )(seed)

    np.testing.assert_allclose(result.x, jax.lax.complex(x, jnp.zeros_like(x)))
    np.testing.assert_allclose(
        result.dense_jacobian,
        seed.dense_jacobian.astype(jnp.complex64),
    )
    np.testing.assert_allclose(
        result.laplacian,
        jnp.zeros_like(x, dtype=jnp.complex64),
    )
