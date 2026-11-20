# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Custom Forward Laplacian rule tests."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from jaqmc.laplacian import (
    LapTuple,
    custom_laplacian,
    forward_laplacian,
)
from tests.laplacian.helpers import assert_allclose


class TestCustomLaplacian:
    def test_called_outside_forward_laplacian(self):
        """Decorated function works normally outside forward_laplacian."""

        @custom_laplacian
        def my_sin(x):
            return jnp.sin(x)

        @my_sin.def_laplacian_rule
        def _(x):
            return LapTuple(
                jnp.sin(x.x),
                jnp.cos(x.x) * x.jacobian,
                jnp.cos(x.x) * x.laplacian - jnp.sin(x.x) * (x.jacobian**2).sum(axis=0),
            )

        x = jnp.array([1.0, 2.0])
        assert_allclose(my_sin(x), jnp.sin(x))

    def test_kwargs_work_outside_forward_laplacian(self):
        """Decorated functions keep their ordinary Python call signature."""

        @custom_laplacian
        def shifted_square(x, *, shift=0.0):
            return (x + shift) ** 2

        x = jnp.array([1.0, 2.0])
        assert_allclose(shifted_square(x, shift=3.0), (x + 3.0) ** 2)

    def test_kwargs_raise_inside_forward_laplacian(self):
        """The custom primitive path is positional-only."""

        @custom_laplacian
        def shifted_square(x, *, shift=0.0):
            return (x + shift) ** 2

        @shifted_square.def_laplacian_rule
        def _(x):
            return LapTuple(
                x.x**2,
                2 * x.x * x.jacobian,
                2 * x.x * x.laplacian + 2 * (x.jacobian**2).sum(axis=0),
            )

        fl = forward_laplacian(lambda x: jnp.sum(shifted_square(x, shift=3.0)))
        with pytest.raises(TypeError, match=r"positional arguments only.*shift"):
            fl(jnp.array([1.0, 2.0]))

    def test_preserves_function_name(self):
        """functools.update_wrapper preserves __name__."""

        @custom_laplacian
        def fancy_name(x):
            return x

        assert fancy_name.__name__ == "fancy_name"

    def test_warns_on_mutable_rule_closure(self):
        """Rules closing over mutable Python state emit a warning."""
        state = {"scale": 2.0}

        @custom_laplacian
        def scaled_square(x):
            return x**2

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @scaled_square.def_laplacian_rule
            def _(x):
                scale = state["scale"]
                return LapTuple(
                    scale * x.x**2,
                    scale * 2 * x.x * x.jacobian,
                    scale * (2 * x.x * x.laplacian + 2 * (x.jacobian**2).sum(axis=0)),
                )

        assert any(
            "closes over nontrivial Python state" in str(w.message) for w in caught
        )


class TestCustomLaplacianPublicTransforms:
    @staticmethod
    def _make_square():
        @custom_laplacian
        def my_square(x):
            return x**2

        return my_square

    def test_custom_laplacian_works_like_plain_outside(self):
        """Outside forward_laplacian, JVP differentiates the original function."""
        my_square = self._make_square()
        x = jnp.array([1.0, 2.0, 3.0])
        t = jnp.ones_like(x)
        primals_out, tangents_out = jax.jvp(my_square, (x,), (t,))
        assert_allclose(primals_out, x**2)
        assert_allclose(tangents_out, 2 * x)

    def test_custom_rule_is_independent_of_primal_jit_call_order(self) -> None:
        """Primal and derivative traces must not reuse each other's cached graph."""

        def make_square():
            @custom_laplacian
            def square(value: jnp.ndarray) -> jnp.ndarray:
                # Only the custom Laplacian rule differentiates this primal.
                return jax.lax.stop_gradient(value**2)

            @square.def_laplacian_rule
            def square_rule(value: LapTuple[jnp.ndarray]) -> LapTuple[jnp.ndarray]:
                return LapTuple(
                    value.x**2,
                    2 * value.x * value.dense_jacobian,
                    2 * value.x * value.laplacian
                    + 2 * jnp.sum(value.dense_jacobian**2, axis=0),
                )

            return jax.jit(square), jax.jit(forward_laplacian(square))

        x = jnp.array([0.25, -1.5], dtype=jnp.float32)
        for primal_first in (False, True):
            primal, derivative = make_square()
            if primal_first:
                primal(x)
            else:
                derivative(x)
            actual = derivative(x)
            assert isinstance(actual, LapTuple)
            assert_allclose(primal(x), x**2)
            assert_allclose(actual.x, x**2)
            assert_allclose(actual.dense_jacobian, jnp.diag(2 * x))
            assert_allclose(actual.laplacian, jnp.full_like(x, 2))
            assert_allclose(
                jax.jvp(primal, (x,), (jnp.ones_like(x),))[1], jnp.zeros_like(x)
            )

    def test_custom_laplacian_jits_like_plain_function(self):
        """Outside forward_laplacian, JIT lowers the primal custom call."""
        my_square = self._make_square()
        x = jnp.array([1.0, 2.0, 3.0])

        assert_allclose(jax.jit(my_square)(x), x**2)
