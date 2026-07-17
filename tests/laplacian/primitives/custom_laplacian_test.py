# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""custom_laplacian primitive-handler behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    AutoLaplacianFallback,
    LapTuple,
    custom_laplacian,
    forward_laplacian,
    make_laplacian_input,
)
from tests.laplacian.helpers import assert_allclose, check_with_brute_force
from tests.laplacian.input_fixtures import tracked_case_input


class TestCustomLaplacianPrimitive:
    def test_basic_with_rule(self):
        """Custom rule for x^2 matches brute-force Laplacian."""

        @custom_laplacian
        def my_square(x):
            return x**2

        @my_square.def_laplacian_rule
        def _(x):
            val = x.x**2
            jac = 2 * x.x * x.jacobian
            lapl = 2 * x.x * x.laplacian + 2 * (x.jacobian**2).sum(axis=0)
            return LapTuple(val, jac, lapl)

        x = jnp.array([1.0, 2.0, 3.0])
        fn = lambda x: jnp.sum(my_square(x))
        check_with_brute_force(fn, x)

    @pytest.mark.parametrize(
        "case",
        (
            pytest.param("local1", id="local1"),
            pytest.param("local2", id="local2"),
        ),
    )
    def test_rule_can_use_dense_jacobian_with_sparse_seed(self, case):
        """Generic custom rules may rely on ``dense_jacobian`` under sparse seeding."""

        @custom_laplacian
        def my_square(x):
            return x**2

        @my_square.def_laplacian_rule
        def _(x):
            dense_jacobian = x.dense_jacobian
            val = x.x**2
            jac = 2 * x.x * dense_jacobian
            lapl = 2 * x.x * x.laplacian + 2 * (dense_jacobian**2).sum(axis=0)
            return LapTuple(val, jac, lapl)

        seed: jnp.ndarray | LapTuple
        if case == "local1":
            seed = make_laplacian_input(
                jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
                sparse_axis=0,
            )
        else:
            seed = tracked_case_input(
                jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3),
                "local2",
            )
        fn = lambda z: jnp.sum(my_square(z))
        check_with_brute_force(fn, seed)

    def test_without_rule_raises(self):
        """Using custom_laplacian without a rule raises inside forward_laplacian."""

        @custom_laplacian
        def my_cube(x):
            return x**3

        x = jnp.array([1.0, 2.0])
        fn = lambda x: jnp.sum(my_cube(x))
        fl = forward_laplacian(fn)
        with np.testing.assert_raises_regex(
            ValueError, "without a registered def_laplacian_rule"
        ):
            fl(x)

    def test_inside_larger_computation(self):
        """Custom-laplacian function composed with other ops."""

        @custom_laplacian
        def my_exp(x):
            return jnp.exp(x)

        @my_exp.def_laplacian_rule
        def _(x):
            e = jnp.exp(x.x)
            jac = e * x.jacobian
            lapl = e * x.laplacian + e * (x.jacobian**2).sum(axis=0)
            return LapTuple(e, jac, lapl)

        x = jnp.array([0.5, -0.3, 1.2])
        fn = lambda x: jnp.sum(my_exp(x) * x)
        check_with_brute_force(fn, x)

    def test_rule_called_not_fallback(self):
        """Verify the custom rule is actually invoked, not the fallback."""

        @custom_laplacian
        def tracked_fn(x):
            return x**2

        @tracked_fn.def_laplacian_rule
        def _(x):
            return LapTuple(
                x.x**2,
                2 * x.x * x.jacobian,
                jnp.full_like(x.x, -123.0),
            )

        x = jnp.array([1.0, 2.0])
        fl = forward_laplacian(lambda x: jnp.sum(tracked_fn(x)))
        result = fl(x)
        assert_allclose(result.laplacian, -246.0)

    def test_rule_can_delegate_to_auto_dense_path(self):
        """Custom rules may defer unsupported cases to the dense auto path."""

        @custom_laplacian
        def my_exp(x):
            return jnp.exp(x)

        rule_calls = 0

        @my_exp.def_laplacian_rule
        def _(x):
            nonlocal rule_calls
            rule_calls += 1
            raise AutoLaplacianFallback("test explicit dense delegation")

        x = jnp.array([0.5, -0.3, 1.2])
        fn = lambda x: jnp.sum(my_exp(x) * x)
        fl = forward_laplacian(fn)
        result = fl(x)
        assert rule_calls == 1
        check_with_brute_force(fn, x, actual_result=result)

    def test_rule_fallback_preserves_sparse_input_correctness(self):
        """Auto fallback replays the ordinary sparse-aware interpreter path."""

        @custom_laplacian
        def my_exp(x):
            return jnp.exp(x)

        @my_exp.def_laplacian_rule
        def _(x):
            raise AutoLaplacianFallback("test sparse delegation")

        seed = make_laplacian_input(
            jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3) / 10.0,
            sparse_axis=0,
        )
        fn = lambda x: jnp.sum(my_exp(x) * x)

        check_with_brute_force(fn, seed)

    def test_pytree_output_splits_leaves_and_propagates_nonlinear_leaf(self):
        """Tuple outputs with differently shaped leaves and a custom nonlinear rule."""

        @custom_laplacian
        def split_and_square_fn(x):
            return x[:2], x[2:] ** 2

        @split_and_square_fn.def_laplacian_rule
        def _(x):
            lhs = LapTuple(x.x[:2], x.jacobian[:, :2], x.laplacian[:2])
            squared = LapTuple(
                x.x[2:] ** 2,
                2 * x.x[2:] * x.jacobian[:, 2:],
                2 * x.x[2:] * x.laplacian[2:]
                + 2 * (x.jacobian[:, 2:] ** 2).sum(axis=0),
            )
            return lhs, squared

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fn = lambda x: (
            jnp.sum(split_and_square_fn(x)[0]) + jnp.sum(split_and_square_fn(x)[1])
        )
        check_with_brute_force(fn, x)

    def test_pytree_input(self):
        """Function receiving a dict of arrays."""

        @custom_laplacian
        def dict_fn(d):
            return d["a"] * d["b"]

        @dict_fn.def_laplacian_rule
        def _(d):
            val = d["a"].x * d["b"].x
            jac = d["b"].x * d["a"].jacobian + d["a"].x * d["b"].jacobian
            lapl = (
                d["b"].x * d["a"].laplacian
                + d["a"].x * d["b"].laplacian
                + 2 * (d["a"].jacobian * d["b"].jacobian).sum(axis=0)
            )
            return LapTuple(val, jac, lapl)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        fn = lambda x: jnp.sum(dict_fn({"a": x[:2], "b": x[2:]}))
        check_with_brute_force(fn, x)

    def test_jvp_of_custom_laplacian_uses_primal_derivative(self):
        """JAX JVP through a custom primitive remains observable to the transform."""

        @custom_laplacian
        def my_square(x):
            return x**2

        @my_square.def_laplacian_rule
        def _(x):
            return LapTuple(
                x.x**2,
                2 * x.x * x.dense_jacobian,
                2 * x.x * x.laplacian + 2 * jnp.sum(x.dense_jacobian**2, axis=0),
            )

        def fn(x):
            primal, tangent = jax.jvp(
                my_square,
                (x,),
                (jnp.ones_like(x),),
            )
            return jnp.sum(primal + tangent)

        x = jnp.array([0.2, -0.3], dtype=jnp.float32)
        check_with_brute_force(fn, x)


class TestCustomLaplacianPrimitiveReplay:
    """custom_laplacian primitive replay through surrounding JAX transforms."""

    @staticmethod
    def _make_square():
        @custom_laplacian
        def my_square(x):
            return x**2

        return my_square

    @staticmethod
    def _attach_constant_laplacian_rule(fn, constant):
        @fn.def_laplacian_rule
        def _(x):
            return LapTuple(
                x.x**2,
                2 * x.x * x.jacobian,
                jnp.full_like(x.x, constant),
            )

        return fn

    def test_vmap_rule_can_delegate_to_auto_dense_path(self):
        """Vmapped custom rules may still delegate to the dense auto path."""

        @custom_laplacian
        def my_square(x):
            return x**2

        rule_calls = 0

        @my_square.def_laplacian_rule
        def _(x):
            nonlocal rule_calls
            rule_calls += 1
            raise AutoLaplacianFallback("test vmapped dense delegation")

        x = jnp.arange(6.0) / 10.0
        fn = lambda x: jnp.sum(jax.vmap(my_square)(x.reshape(3, 2)))
        fl = forward_laplacian(fn)
        result = fl(x)
        assert rule_calls == 1
        check_with_brute_force(fn, x, actual_result=result)

    def test_vmap_before_forward_laplacian_uses_custom_rule(self):
        """forward_laplacian uses the custom rule after vmapping first."""
        my_square = self._attach_constant_laplacian_rule(self._make_square(), -123.0)
        vmapped = jax.vmap(my_square)
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        fl = forward_laplacian(lambda x: jnp.sum(vmapped(x)))
        result = fl(x)

        assert_allclose(result.laplacian, -123.0 * x.size)

    def test_nested_vmap_before_forward_laplacian_uses_custom_rule(self):
        """Nested vmaps replay the custom rule around every mapped axis."""
        my_square = self._attach_constant_laplacian_rule(self._make_square(), -123.0)
        vmapped = jax.vmap(jax.vmap(my_square))
        x = jnp.arange(8.0, dtype=jnp.float32).reshape(2, 2, 2)

        result = forward_laplacian(lambda value: jnp.sum(vmapped(value)))(x)

        assert_allclose(result.laplacian, -123.0 * x.size)

    def test_jit_vmap_before_forward_laplacian_uses_custom_rule(self):
        """The custom rule survives jit(vmap(fn)) before forward_laplacian."""
        my_square = self._attach_constant_laplacian_rule(self._make_square(), -123.0)
        transformed = jax.jit(jax.vmap(my_square))
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        fl = forward_laplacian(lambda x: jnp.sum(transformed(x)))
        result = fl(x)

        assert_allclose(result.laplacian, -123.0 * x.size)

    def test_scan_body_uses_custom_rule(self):
        """A scan body calling custom_laplacian uses the custom rule."""
        my_square = self._attach_constant_laplacian_rule(self._make_square(), 123.0)

        def fn(x):
            def body(carry, _):
                return carry + my_square(carry), None

            final, _ = jax.lax.scan(body, x, None, length=2)
            return jnp.sum(final)

        x = jnp.array([0.2, -0.4])
        fl = forward_laplacian(fn)
        result = fl(x)

        assert_allclose(result.laplacian, 246.0 * x.size)

    def test_scan_body_uses_vmapped_custom_rule(self):
        """A scan body using vmap(custom_laplacian) keeps the custom rule."""
        my_square = self._attach_constant_laplacian_rule(self._make_square(), 123.0)
        vmapped_square = jax.vmap(my_square)

        def fn(x):
            def body(carry, _):
                return carry + vmapped_square(carry), None

            final, _ = jax.lax.scan(body, x, None, length=2)
            return jnp.sum(final)

        x = jnp.array([[0.2, -0.4], [0.1, 0.3]])
        fl = forward_laplacian(fn)
        result = fl(x)

        assert_allclose(result.laplacian, 246.0 * x.size)
