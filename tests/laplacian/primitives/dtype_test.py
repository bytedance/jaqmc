# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""convert_element_type semantics: float dtypes keep tracking, others drop."""

import jax
import jax.numpy as jnp
import pytest

from jaqmc.laplacian import LapTuple, forward_laplacian
from tests.laplacian.helpers import assert_allclose, check_with_brute_force
from tests.laplacian.input_fixtures import (
    VECTOR_SHAPE,
    random_array,
    to_complex,
    tracked_case_input,
)

from .helpers import parametrize_over_tracked_cases


@parametrize_over_tracked_cases("case")
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float16,
        jnp.float32,
        pytest.param(jnp.float64, marks=pytest.mark.requires_x64),
    ],
)
def test_float_dtypes_preserve_tracking(case, dtype):
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)
    packed = random_array("real")
    rtol = 1e-3 if dtype == jnp.float16 else 1e-5
    seed = tracked_case_input(packed, case)
    result = forward_laplacian(
        lambda value: jax.lax.convert_element_type(value, dtype)
    )(seed)
    check_with_brute_force(
        lambda value: jax.lax.convert_element_type(value, dtype),
        seed,
        actual_result=result,
        rtol=rtol,
        atol=1e-3 if dtype == jnp.float16 else 1e-10,
    )
    assert isinstance(result, LapTuple)
    assert result.x.dtype == expected_dtype
    assert result.dense_jacobian.dtype == expected_dtype
    assert result.laplacian.dtype == expected_dtype


@parametrize_over_tracked_cases("case")
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.complex64,
        pytest.param(jnp.complex128, marks=pytest.mark.requires_x64),
    ],
)
def test_complex_dtypes_preserve_tracking(case, dtype):
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)
    packed = random_array("real")
    seed = tracked_case_input(packed, case)
    result = forward_laplacian(
        lambda value: jax.lax.convert_element_type(value, dtype)
    )(seed)
    check_with_brute_force(
        lambda value: jax.lax.convert_element_type(value, dtype),
        seed,
        actual_result=result,
    )
    assert isinstance(result, LapTuple)
    assert result.x.dtype == expected_dtype
    assert result.dense_jacobian.dtype == expected_dtype
    assert result.laplacian.dtype == expected_dtype


@parametrize_over_tracked_cases("case")
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.bool_,
        jnp.int8,
        jnp.int16,
        jnp.int32,
        pytest.param(jnp.int64, marks=pytest.mark.requires_x64),
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        pytest.param(jnp.uint64, marks=pytest.mark.requires_x64),
    ],
)
def test_non_float_dtypes_drop_tracking(case, dtype):
    fl = forward_laplacian(lambda v: jax.lax.convert_element_type(v, dtype))
    seed = tracked_case_input(random_array("real"), case)
    assert isinstance(seed, LapTuple)
    result = fl(seed)
    assert not isinstance(result, LapTuple)
    assert isinstance(result, jax.Array)
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)
    assert result.dtype == expected_dtype
    assert jnp.array_equal(result, seed.x.astype(expected_dtype))


@parametrize_over_tracked_cases("case")
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.complex64,
        pytest.param(jnp.complex128, marks=pytest.mark.requires_x64),
    ],
)
def test_complex_input_complex_cast_preserves_tracking(case, dtype):
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)
    packed = random_array("real", shape=VECTOR_SHAPE, key=9)
    seed = tracked_case_input(packed, case, key=10)
    assert isinstance(seed, LapTuple)
    check_with_brute_force(
        lambda value: jax.lax.convert_element_type(to_complex(value), dtype),
        seed,
    )
    result = forward_laplacian(
        lambda value: jax.lax.convert_element_type(to_complex(value), dtype)
    )(seed)
    assert isinstance(result, LapTuple)
    assert result.x.dtype == expected_dtype
    assert result.dense_jacobian.dtype == expected_dtype
    assert result.laplacian.dtype == expected_dtype
    assert_allclose(result.x, to_complex(seed.x).astype(expected_dtype))


@parametrize_over_tracked_cases("case")
@pytest.mark.parametrize(
    "dtype",
    [
        jnp.bool_,
        jnp.int32,
        jnp.uint8,
    ],
)
def test_complex_input_non_float_cast_drops_tracking(case, dtype):
    # JAX deprecates casting complex directly to noncomplex dtypes; use
    # jnp.real or jnp.imag before convert_element_type. Real-projection
    # coverage lives in complex_test.py.
    packed = random_array("real", shape=VECTOR_SHAPE, key=11)

    def cast_real_part(value):
        return jax.lax.convert_element_type(jnp.real(to_complex(value)), dtype)

    fl = forward_laplacian(cast_real_part)
    seed = tracked_case_input(packed, case, key=12)
    assert isinstance(seed, LapTuple)
    result = fl(seed)
    assert not isinstance(result, LapTuple)
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)
    assert result.dtype == expected_dtype
    assert_allclose(result, cast_real_part(seed.x))
