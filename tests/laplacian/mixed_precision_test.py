# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward Laplacian contraction scopes and scaled-FP16 kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    LapTuple,
    Local1Jacobian,
    Local2Jacobian,
    forward_laplacian,
    make_laplacian_input,
)
from jaqmc.laplacian.precision import (
    ContractionMode,
    jacobian_contraction_scope,
    scaled_dot_general,
    scaled_einsum,
)


def test_nested_contraction_scopes_are_call_order_invariant() -> None:
    """An inner standard scope overrides scaled FP16, including cached JIT calls."""
    x = jnp.array([[0.123456, -0.731245], [1.234567, 0.431287]], dtype=jnp.float32)
    basis = jnp.eye(x.size, dtype=x.dtype).reshape(x.size, *x.shape)
    rounded_x = x.astype(jnp.float16).astype(jnp.float32)
    standard_jac = jax.vmap(lambda d: d @ x.T + x @ d.T)(basis)
    scaled_jac = jax.vmap(lambda d: d @ rounded_x.T + rounded_x @ d.T)(basis)
    assert jnp.any(standard_jac != scaled_jac)

    for primal_first in (False, True):

        @jax.jit
        def gram(value: jnp.ndarray) -> jnp.ndarray:
            return value @ value.T

        @jax.jit
        def fn(value: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
            outside = gram(value)
            with jacobian_contraction_scope(ContractionMode.SCALED_FP16):
                scaled = gram(value)
                with jacobian_contraction_scope(ContractionMode.STANDARD):
                    standard = gram(value)
                restored = gram(value)
            return outside, scaled, standard, restored

        derivative = jax.jit(forward_laplacian(fn))
        if primal_first:
            fn(x)
        else:
            derivative(x)
        actual = derivative(x)
        values = fn(x)
        for result, value, expected_jac in zip(
            actual,
            values,
            (standard_jac, scaled_jac, standard_jac, scaled_jac),
            strict=True,
        ):
            np.testing.assert_array_equal(result.x, value)
            np.testing.assert_array_equal(result.dense_jacobian, expected_jac)
            np.testing.assert_array_equal(result.laplacian, 4 * jnp.eye(2))


def test_scaled_linear_dot_keeps_slice_scaling_zeros_and_fp32_laplacian() -> None:
    x = jnp.array([[0.312345, 0.456789], [-0.178912, 0.923456]], dtype=jnp.float32)
    weight = jnp.array([[1.234567, -0.731245], [0.431287, 1.678912]], dtype=jnp.float32)
    payload = jnp.array(
        [[0.912345, 0.456789], [-0.731245, 1.234567]], dtype=jnp.float32
    )
    jac = jnp.stack([jnp.zeros_like(payload), payload * 2.0**-12, payload * 2.0**18])
    lap = x * 0.7
    seed = LapTuple(x, jac, lap)

    for tracked_is_lhs in (False, True):

        def fn(value: jnp.ndarray, lhs_tracked: bool = tracked_is_lhs) -> jnp.ndarray:
            lhs, rhs = (value, weight) if lhs_tracked else (weight, value)
            with jacobian_contraction_scope(ContractionMode.SCALED_FP16):
                return jnp.matmul(lhs, rhs, precision=jax.lax.Precision.DEFAULT)

        actual = jax.jit(forward_laplacian(fn))(seed)
        lhs, rhs = (x, weight) if tracked_is_lhs else (weight, x)
        expected_x = jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)
        expected_lap = jnp.matmul(
            lap if tracked_is_lhs else weight,
            weight if tracked_is_lhs else lap,
            precision=jax.lax.Precision.HIGHEST,
        )
        expected_jac = jax.vmap(lambda d: d @ weight if tracked_is_lhs else weight @ d)(
            jac
        )
        assert (
            actual.x.dtype
            == actual.dense_jacobian.dtype
            == actual.laplacian.dtype
            == jnp.float32
        )
        np.testing.assert_array_equal(actual.x, expected_x)
        np.testing.assert_array_equal(actual.laplacian, expected_lap)
        np.testing.assert_array_equal(
            actual.dense_jacobian[0], jnp.zeros_like(expected_jac[0])
        )
        assert jnp.all(actual.dense_jacobian[1] != 0)
        np.testing.assert_allclose(actual.dense_jacobian, expected_jac, rtol=2e-3)
        assert jnp.any(actual.dense_jacobian != expected_jac)


def test_scaled_fp16_handles_subnormal_and_max_finite_fp32_slices() -> None:
    """Extreme finite FP32 slices (tiny, subnormal, and max) do not overflow scaling."""
    ftiny = float(jnp.finfo(jnp.float32).tiny)
    fmax = float(jnp.finfo(jnp.float32).max)
    lhs = jnp.array(
        [
            [2.0 * ftiny, 0.0],
            [1e-40, 0.0],
            [fmax, 0.0],
        ],
        dtype=jnp.float32,
    )
    rhs = jnp.array(
        [
            [0.5, ftiny, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    dn = (((1,), (0,)), ((), ()))
    actual = scaled_dot_general(lhs, rhs, dn, mode=ContractionMode.SCALED_FP16)
    expected = scaled_dot_general(lhs, rhs, dn, mode=ContractionMode.STANDARD)
    assert jnp.all(jnp.isfinite(actual))
    assert actual[0, 0] > 0
    assert actual[2, 0] > 0
    assert actual[2, 1] > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-3)


def test_sparse_linear_dot_matches_dense_per_owner_scaling() -> None:
    """Sparse blocks scale per (coord, owner) slice and match dense scaling."""
    x = jnp.array(
        [[[0.312345, -0.456789], [0.712345, 0.234567]]],
        dtype=jnp.float32,
    )
    weight = jnp.array([[1.234567, -0.731245], [0.431287, 1.678912]], dtype=jnp.float32)
    base = make_laplacian_input(x, sparse_axis=1)
    assert isinstance(base.jacobian, Local1Jacobian)
    owner_scales = jnp.array([2.0**-18, 2.0**18], dtype=jnp.float32)
    blocks = (
        jnp.array([1.123456, -0.876543], dtype=jnp.float32)[:, None, None, None]
        * owner_scales[None, None, :, None]
        * jnp.ones((1, 2, 1, 2, 2), dtype=jnp.float32)
    )
    sparse_seed = LapTuple(x, base.jacobian.with_blocks(blocks), jnp.zeros_like(x))
    dense_seed = sparse_seed.to_dense()

    for lhs_tracked in (True, False):

        def fn(value: jnp.ndarray, is_lhs: bool = lhs_tracked) -> jnp.ndarray:
            with jacobian_contraction_scope(ContractionMode.SCALED_FP16):
                if is_lhs:
                    return jnp.einsum("bni,ij->bnj", value, weight)
                return jnp.einsum("ij,bnj->ibn", weight, value)

        sparse_out = jax.jit(forward_laplacian(fn))(sparse_seed)
        dense_out = jax.jit(forward_laplacian(fn))(dense_seed)
        assert isinstance(sparse_out.jacobian, Local1Jacobian)
        assert jnp.all(sparse_out.jacobian.blocks != 0)
        np.testing.assert_array_equal(
            sparse_out.dense_jacobian, dense_out.dense_jacobian
        )
        np.testing.assert_array_equal(sparse_out.laplacian, dense_out.laplacian)


def test_bilinear_trace_uses_per_slice_scaling() -> None:
    """Per-slice scaling preserves mixed derivative traces across dynamic ranges."""
    x = jnp.array([[0.25, -0.5, 0.75], [0.5, 0.25, -0.25]], dtype=jnp.float32)
    y = jnp.array([[1.0, -0.25, 0.5], [-0.5, 0.75, 0.25]], dtype=jnp.float32)
    lhs = make_laplacian_input(x, sparse_axis=0)
    rhs = make_laplacian_input(y, sparse_axis=0)
    assert isinstance(lhs.jacobian, Local1Jacobian)
    assert isinstance(rhs.jacobian, Local1Jacobian)
    lhs_amplitudes = jnp.array([2.0**-12, 2.0**18, 1.0], dtype=x.dtype)
    rhs_amplitudes = jnp.array([2.0**18, 2.0**-12, 1.0], dtype=y.dtype)
    lhs = LapTuple(
        x,
        lhs.jacobian.with_blocks(
            jnp.broadcast_to(lhs_amplitudes[None, :, None, None], (1, 3, *x.shape))
        ),
        jnp.zeros_like(x),
    )
    rhs = LapTuple(
        y,
        rhs.jacobian.with_blocks(
            jnp.broadcast_to(rhs_amplitudes[None, :, None, None], (1, 3, *y.shape))
        ),
        jnp.zeros_like(y),
    )
    lhs_dense, rhs_dense = lhs.to_dense(), rhs.to_dense()

    def fn(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        with jacobian_contraction_scope(ContractionMode.SCALED_FP16):
            return left @ right.T

    expected_jac = jax.vmap(lambda left_j, right_j: left_j @ y.T + x @ right_j.T)(
        lhs_dense.jacobian, rhs_dense.jacobian
    )
    expected_lap = (2 * (2.0**6 + 2.0**6 + 1.0) * 3) * jnp.eye(2, dtype=x.dtype)
    for sparse, args in ((False, (lhs_dense, rhs_dense)), (True, (lhs, rhs))):
        actual = forward_laplacian(fn)(*args)
        if sparse:
            assert isinstance(actual.jacobian, Local2Jacobian)
        assert actual.x.dtype == actual.dense_jacobian.dtype == actual.laplacian.dtype
        np.testing.assert_array_equal(actual.x, x @ y.T)
        np.testing.assert_allclose(actual.dense_jacobian, expected_jac, rtol=1e-6)
        np.testing.assert_array_equal(actual.laplacian, expected_lap)


def test_scaled_fp16_complex_operands_preserve_complex_derivatives() -> None:
    """Complex operands bypass FP16 casting inside SCALED_FP16 scopes."""
    x = jnp.array([[0.3, -0.4], [0.5, 0.2]], dtype=jnp.float32)
    weight = jnp.array(
        [[1.2 + 0.7j, -0.5 + 0.3j], [0.4 - 0.9j, 0.8 + 0.6j]],
        dtype=jnp.complex64,
    )

    def fn(value: jnp.ndarray, *, mode: ContractionMode) -> jnp.ndarray:
        z = (value + 1j * jnp.flip(value, axis=-1)) ** 2
        with jacobian_contraction_scope(mode):
            return (z @ weight) @ jnp.swapaxes(z, -1, -2)

    seed = make_laplacian_input(x, sparse_axis=0)
    actual = jax.jit(forward_laplacian(fn, mode=ContractionMode.SCALED_FP16))(seed)
    expected = jax.jit(forward_laplacian(fn, mode=ContractionMode.STANDARD))(seed)
    assert isinstance(actual.jacobian, Local2Jacobian)
    assert (
        actual.x.dtype
        == actual.dense_jacobian.dtype
        == actual.laplacian.dtype
        == jnp.complex64
    )
    np.testing.assert_allclose(actual.x, expected.x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual.dense_jacobian, expected.dense_jacobian, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        actual.laplacian, expected.laplacian, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "subscripts",
    [
        "ij,kj->",
        "ij,jk->i",
        "ii,jk->jk",
        "ij,jk->kk",
        "ij,jk->iz",
        "...ij,jk->...ik",
        "ij,jk",
        "ijk,jk->ik",
    ],
)
def test_scaled_einsum_rejects_unsupported_subscripts(subscripts: str) -> None:
    """Reject single-side reductions, traces, ellipsis, and rank mismatches."""
    lhs = jnp.ones((2, 3), dtype=jnp.float32)
    rhs = jnp.ones((3, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        scaled_einsum(subscripts, lhs, rhs)
