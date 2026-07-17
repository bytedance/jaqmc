# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Registry and extensibility tests for Forward Laplacian handlers."""

import logging

import jax
import jax.numpy as jnp
import pytest
from jax.extend.core import Primitive

from jaqmc.laplacian import forward_laplacian
from jaqmc.laplacian import primitives as laplacian_primitives
from jaqmc.laplacian.primitives import (
    deregister_function,
    get_laplacian,
    register_function,
)
from jaqmc.laplacian.primitives.registry import get_laplacian as get_laplacian_impl
from tests.laplacian.helpers import check_with_brute_force


def test_register_custom_handler():
    """Register a custom handler and verify that the registry uses it."""
    call_count = [0]
    original = get_laplacian(jax.lax.exp_p)
    assert original is not None

    def custom_exp(args, kwargs):
        call_count[0] += 1
        return original(args, kwargs)

    register_function(jax.lax.exp_p, custom_exp)
    try:
        x = jnp.array([1.0, 2.0])
        fn = lambda value: jnp.sum(jnp.exp(value))
        result = forward_laplacian(fn)(x)
        assert call_count[0] > 0
        check_with_brute_force(fn, x, actual_result=result)
    finally:
        register_function(jax.lax.exp_p, original)


def test_deregister_and_reregister():
    original = get_laplacian(jax.lax.exp_p)
    assert original is not None
    deregister_function(jax.lax.exp_p)
    try:
        assert get_laplacian(jax.lax.exp_p) is None
    finally:
        register_function(jax.lax.exp_p, original)
    assert get_laplacian(jax.lax.exp_p) is not None


def test_register_and_deregister_by_name():
    dummy = lambda args, kwargs: args[0]
    register_function("test_op", dummy)
    try:
        assert get_laplacian("test_op") is not None
    finally:
        deregister_function("test_op")
    assert get_laplacian("test_op") is None


def test_wrap_if_missing_rejects_string_names():
    with pytest.raises(
        TypeError, match="Can't wrap missing_op based on function names"
    ):
        get_laplacian_impl("missing_op", wrap_if_missing=True)


def test_wrap_if_missing_returns_general_fallback_for_primitives(caplog):
    primitive = Primitive("unregistered_laplacian_test_primitive")

    with caplog.at_level(logging.WARNING):
        handler = get_laplacian_impl(primitive, wrap_if_missing=True)

    assert handler is not None
    assert "Full Hessian fallback will be slow" in caplog.text


def test_deregister_missing_key_raises():
    with pytest.raises(KeyError):
        deregister_function("nonexistent_laplacian_handler_key")


def test_pad_uses_generic_fallback():
    assert get_laplacian(jax.lax.pad_p) is None

    def fn(x):
        return jax.lax.pad(x, 0.0, [(0, 1, 0), (0, 0, 0)])

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    check_with_brute_force(fn, x)


def test_import_surface_matches_declared_primitives_all():
    expected = {
        "AutoLaplacianFallback",
        "deregister_function",
        "get_laplacian",
        "register_function",
        "setup_handler",
        "wrap_elementwise",
        "wrap_general",
        "wrap_linear",
        "wrap_multiplication",
        "wrap_without_fwd_laplacian",
    }
    assert set(laplacian_primitives.__all__) == expected
    for name in laplacian_primitives.__all__:
        assert hasattr(laplacian_primitives, name)
