# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Transform behavior tests for Forward Laplacian."""

import jax
import jax.numpy as jnp

from jaqmc.laplacian import LapTuple, forward_laplacian, make_laplacian_input
from tests.laplacian.helpers import (
    assert_allclose,
    check_with_brute_force,
)


class TestCustomJVP:
    def test_custom_jvp_function(self):
        @jax.custom_jvp
        def my_square(x):
            return x**2

        @my_square.defjvp
        def my_square_jvp(primals, tangents):
            (x,), (t,) = primals, tangents
            return x**2, 2 * x * t

        x = jnp.array([1.0, 2.0, 3.0])
        fn = lambda value: jnp.sum(my_square(value))
        check_with_brute_force(fn, x)

    def test_custom_jvp_modified_derivative(self):
        """custom_jvp with a non-standard derivative is respected."""

        @jax.custom_jvp
        def fn(x):
            return jnp.sin(x)

        @fn.defjvp
        def fn_jvp(primals, tangents):
            (x,) = primals
            (x_dot,) = tangents
            return fn(x), 2 * x_dot * jnp.cos(x)

        x = jnp.array([0.5, -0.3, 1.2, 0.8, -1.0, 0.1, 0.9, -0.7, 1.5, -0.2])
        check_with_brute_force(fn, x, rtol=1e-4)

    def test_real_to_complex_with_wide_tracked_basis(self):
        """R->C custom JVP uses the generic Hessian path for K > D."""

        @jax.custom_jvp
        def real_to_complex(value):
            return value + 1j * value**2

        @real_to_complex.defjvp
        def real_to_complex_jvp(primals, tangents):
            (value,), (tangent,) = primals, tangents
            return real_to_complex(value), (1 + 2j * value) * tangent

        x = jnp.array([0.2, -0.3], dtype=jnp.float32)
        jacobian = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=jnp.float32)
        seed = LapTuple(x, jacobian, jnp.zeros_like(x))
        fn = lambda value: jnp.sum(real_to_complex(value))

        actual = forward_laplacian(fn)(seed)
        expected_jacobian = jacobian @ (1 + 2j * x)
        expected_laplacian = 2j * jnp.sum(jacobian**2)

        assert_allclose(actual.x, fn(x))
        assert_allclose(actual.dense_jacobian, expected_jacobian)
        assert_allclose(actual.laplacian, expected_laplacian)

    def test_complex_to_real_with_wide_tracked_basis(self):
        """C->R generic fallback uses HVP contractions for K > D."""
        x = jnp.array(
            [[1.0 + 0.1j, 0.2 - 0.3j], [0.2 + 0.3j, 2.0 - 0.1j]],
            dtype=jnp.complex64,
        )
        jacobian = (
            jnp.arange(20, dtype=jnp.float32).reshape(5, 2, 2) / 10
            + 1j * jnp.arange(20, 40, dtype=jnp.float32).reshape(5, 2, 2) / 10
        )
        seed = LapTuple(x, jacobian, jnp.zeros_like(x))
        fn = lambda value: jnp.sum(jnp.linalg.svd(value, compute_uv=False))

        actual = forward_laplacian(fn)(seed)
        expected_jacobian = jax.vmap(lambda tangent: jax.jvp(fn, (x,), (tangent,))[1])(
            jacobian
        )

        def second_directional(tangent):
            def first_directional(value):
                return jax.jvp(fn, (value,), (tangent,))[1]

            return jax.jvp(first_directional, (x,), (tangent,))[1]

        expected_laplacian = jnp.sum(jax.vmap(second_directional)(jacobian), axis=0)

        assert_allclose(actual.x, fn(x))
        assert_allclose(actual.dense_jacobian, expected_jacobian)
        assert_allclose(actual.laplacian, expected_laplacian)


class TestScan:
    def test_scan_linear(self):
        """Scan with linear body keeps the Laplacian at zero."""

        def fn(x):
            def body(carry, _):
                return carry + 1.0, None

            final, _ = jax.lax.scan(body, jnp.sum(x), None, length=3)
            return final

        x = jnp.array([1.0, 2.0, 3.0])
        result = forward_laplacian(fn)(x)
        assert_allclose(result.x, fn(x))
        assert jnp.allclose(result.laplacian, 0.0)

    def test_scan_preserves_dense_jacobian_layout(self):
        """Scan keeps the dense basis axis before the output shape."""

        def fn(x):
            def body(carry, _):
                return jnp.tanh(0.5 * carry + 0.1), None

            final, _ = jax.lax.scan(body, x, None, length=2)
            return final

        x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3) / 10.0
        result = forward_laplacian(fn)(x)
        expected_jac = jnp.moveaxis(
            jax.jacfwd(fn)(x).reshape(*result.x.shape, x.size), -1, 0
        )

        assert result.dense_jacobian.shape == (x.size, *result.x.shape)
        assert_allclose(result.dense_jacobian, expected_jac)

    def test_scan_nonlinear_carry(self):
        """Scan with nonlinear carry updates matches the brute-force oracle."""
        W = jnp.array([[0.5, 0.3], [-0.2, 0.8]])

        def fn(x):
            def body(carry, _):
                return jnp.tanh(W @ carry), None

            final, _ = jax.lax.scan(body, x, None, length=3)
            return jnp.sum(final)

        check_with_brute_force(fn, jnp.array([0.5, -0.3]), rtol=1e-4)

    def test_scan_promotes_literal_float_carry(self):
        """Scan promotes an inline float literal before it consumes tracked inputs."""

        def fn(x):
            def body(carry, value):
                return jnp.tanh(carry + value), None

            final, _ = jax.lax.scan(body, 0.0, x)
            return final

        check_with_brute_force(fn, jnp.array([0.2, -0.3]), rtol=1e-4)

    def test_scan_promotes_array_literal_carry(self):
        """Scan promotes an inline array literal before it consumes tracked inputs."""

        def fn(x):
            def body(carry, value):
                return jnp.tanh(carry + value), None

            final, _ = jax.lax.scan(body, jnp.array(0.0), x)
            return final

        check_with_brute_force(fn, jnp.array([0.2, -0.3]), rtol=1e-4)

    def test_scan_promotes_dynamic_plain_carry(self):
        """Scan gives an untracked array carry zero dense derivative state."""

        def fn(x, initial):
            def body(carry, value):
                return jnp.tanh(carry + value), None

            final, _ = jax.lax.scan(body, initial, x)
            return final

        x = make_laplacian_input(jnp.array([0.2, -0.3]))
        check_with_brute_force(fn, x, jnp.array(0.0), rtol=1e-4)

    def test_scan_densifies_sparse_carry(self):
        """Scan densifies sparse carry state; the post-scan payload stays dense."""

        def fn(x):
            def body(carry, _):
                return jnp.tanh(carry + 0.1 * x), None

            final, _ = jax.lax.scan(body, x, None, length=2)
            return jnp.sum(final)

        x = make_laplacian_input(
            jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
            sparse_axis=0,
        )
        check_with_brute_force(fn, x, rtol=1e-4)

        result = forward_laplacian(fn)(x)
        assert isinstance(result.jacobian, jnp.ndarray)

    def test_scan_keeps_integer_and_boolean_carries_plain(self):
        """Scan leaves non-differentiable carries outside derivative state."""

        def state_fn(x):
            def body(carry, value):
                step, active, total = carry
                return (step + 1, active, jnp.tanh(total + value + 0.1 * step)), None

            final, _ = jax.lax.scan(body, (0, True, 0.0), x)
            return final

        x = jnp.array([0.2, -0.3])
        step, active, total = forward_laplacian(state_fn)(x)
        assert not isinstance(step, LapTuple)
        assert not isinstance(active, LapTuple)
        assert isinstance(total, LapTuple)
        assert_allclose(total.x, state_fn(x)[-1])
        check_with_brute_force(lambda value: state_fn(value)[-1], x, rtol=1e-4)


class TestControlFlow:
    def test_cond_matches_brute_force_active_branch(self):
        def fn(x):
            return jax.lax.cond(
                jnp.sum(x) > 0,
                lambda value: jnp.sum(value**2),
                lambda value: jnp.sum(jnp.sin(value)),
                x,
            )

        check_with_brute_force(fn, jnp.array([0.6, -0.2]), rtol=1e-4)

    def test_fori_loop_keeps_integer_counter_untracked(self):
        def fn(x):
            return jax.lax.fori_loop(
                0,
                3,
                lambda i, value: jnp.tanh(value + 0.1 * i),
                x,
            ).sum()

        check_with_brute_force(fn, jnp.array([0.2, -0.3]), rtol=1e-4)

    def test_while_loop_keeps_integer_counter_untracked(self):
        def fn(x):
            def cond_fun(state):
                i, _ = state
                return i < 3

            def body_fun(state):
                i, value = state
                return i + 1, jnp.tanh(value)

            _, value = jax.lax.while_loop(cond_fun, body_fun, (0, x))
            return jnp.sum(value)

        def unrolled_fn(x):
            return jnp.sum(jnp.tanh(jnp.tanh(jnp.tanh(x))))

        check_with_brute_force(
            fn,
            jnp.array([0.2, -0.3], dtype=jnp.float32),
            expected_fn=unrolled_fn,
            rtol=1e-4,
        )

    def test_while_loop_with_consts_and_vector_output(self):
        scale = jnp.array([0.7, -0.2], dtype=jnp.float32)
        bias = jnp.array([0.05, 0.1], dtype=jnp.float32)

        def fn(x):
            def cond_fun(state):
                i, _ = state
                return i < 2

            def body_fun(state):
                i, value = state
                return i + 1, jnp.sin(value * scale + bias)

            _, value = jax.lax.while_loop(cond_fun, body_fun, (0, x))
            return value * scale

        def unrolled_fn(x):
            value = x
            for _ in range(2):
                value = jnp.sin(value * scale + bias)
            return value * scale

        check_with_brute_force(
            fn,
            jnp.array([0.2, -0.3], dtype=jnp.float32),
            expected_fn=unrolled_fn,
            rtol=1e-4,
        )

    def test_sparse_input_through_while_loop_matches_brute_force(self):
        def fn(x):
            def cond_fun(state):
                i, _ = state
                return i < 2

            def body_fun(state):
                i, value = state
                return i + 1, jnp.tanh(value)

            _, value = jax.lax.while_loop(cond_fun, body_fun, (0, x))
            return jnp.sum(value)

        def unrolled_fn(x):
            return jnp.sum(jnp.tanh(jnp.tanh(x)))

        x = jnp.array([[0.2], [-0.3]], dtype=jnp.float32)
        check_with_brute_force(
            fn,
            make_laplacian_input(x, sparse_axis=0),
            expected_fn=unrolled_fn,
            rtol=1e-4,
        )


class TestOuterTransforms:
    def test_jit_forward_laplacian_result(self):
        fn = lambda x: jnp.sum(jnp.sin(x))
        x = jnp.array([0.2, -0.3], dtype=jnp.float32)

        result = jax.jit(forward_laplacian(fn))(x)

        assert_allclose(result.x, fn(x))
        assert_allclose(result.dense_jacobian, jnp.cos(x))
        assert_allclose(result.laplacian, -jnp.sum(jnp.sin(x)))

    def test_inner_jit_matches_brute_force(self):
        """A JIT subcomputation re-enters the Laplacian interpreter correctly."""
        compiled = jax.jit(lambda value: jnp.sum(jnp.sin(value)))
        x = jnp.array([0.2, -0.3], dtype=jnp.float32)

        check_with_brute_force(compiled, x)

    def test_vmap_forward_laplacian_result(self):
        fn = lambda x: jnp.sum(jnp.sin(x))
        x = jnp.array([[0.2, -0.3, 0.6], [0.5, 0.7, -0.4]], dtype=jnp.float32)

        result = jax.vmap(
            forward_laplacian(fn),
            out_axes=LapTuple.pytree_spec(0, 1, 0),
        )(x)

        assert_allclose(result.x, jax.vmap(fn)(x))
        expected_jacobian = jnp.moveaxis(jnp.cos(x), 0, 1)
        assert result.jacobian.shape == expected_jacobian.shape
        assert result.dense_jacobian.shape == (x.shape[1], x.shape[0])
        assert_allclose(result.jacobian, expected_jacobian)
        assert_allclose(result.dense_jacobian, expected_jacobian)
        assert_allclose(result.laplacian, -jnp.sum(jnp.sin(x), axis=-1))

    def test_vmap_sparse_seed_blocks_places_batch_after_prefix(self):
        def fn(x):
            seed = make_laplacian_input(x, sparse_axis=0)
            return seed.jacobian.blocks

        x = jnp.stack(
            [
                jnp.array([[0.2, -0.3], [0.5, 0.7]], dtype=jnp.float32),
                jnp.array([[0.1, 0.4], [0.6, 0.8]], dtype=jnp.float32),
            ]
        )
        batched_blocks = jax.vmap(fn, out_axes=2)(x)
        per_example = jnp.stack([fn(example) for example in x], axis=2)

        assert batched_blocks.shape == per_example.shape
        assert batched_blocks.shape[:2] == (1, 2)
        assert_allclose(batched_blocks, per_example)

    def test_vmap_forward_laplacian_plain_result(self):
        """Vmapped extracted arrays use JAX's default output axis."""
        fn = lambda x: jnp.sum(jnp.sin(x))
        x = jnp.array([[0.2, -0.3, 0.6], [0.5, 0.7, -0.4]], dtype=jnp.float32)

        laplacian = jax.vmap(lambda value: forward_laplacian(fn)(value).laplacian)(x)

        assert laplacian.shape == (x.shape[0],)
        assert_allclose(laplacian, -jnp.sum(jnp.sin(x), axis=-1))
