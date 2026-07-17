# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Flax integration tests for Forward Laplacian."""

import warnings

import jax
import jax.numpy as jnp
from flax import linen as nn

from jaqmc.laplacian import LapTuple, custom_laplacian, forward_laplacian
from tests.laplacian.helpers import assert_allclose, check_with_brute_force


class TestFlaxModuleApply:
    def test_linen_apply_matches_brute_force(self):
        class TinyModule(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=4)(x)
                x = jnp.tanh(x)
                x = nn.Dense(features=1)(x)
                return jnp.sum(x)

        module = TinyModule()
        x = jnp.array([0.5, -0.3, 1.2], dtype=jnp.float32)
        params = module.init(jax.random.PRNGKey(0), x)

        fn = lambda value: module.apply(params, value)
        check_with_brute_force(fn, x, rtol=1e-4)


class TestFlaxCustomRuleRegistration:
    def test_linen_apply_wrapper_uses_custom_rule(self):
        class SquareModule(nn.Module):
            scale: float = 2.0

            def square(self, x):
                return self.scale * x**2

            def square_rule(self, x):
                scale = self.scale
                return LapTuple(
                    scale * x.x**2,
                    scale * 2 * x.x * x.jacobian,
                    jnp.full_like(x.x, -7.0),
                )

        module = SquareModule()
        params = module.init(
            jax.random.PRNGKey(0),
            jnp.array([1.0], dtype=jnp.float32),
            method=module.square,
        )

        @custom_laplacian
        def square(x):
            return module.apply(params, x, method=module.square)

        @square.def_laplacian_rule
        def _(value):
            return module.apply(params, value, method=module.square_rule)

        result = forward_laplacian(lambda value: jnp.sum(square(value)))(
            jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        )
        assert_allclose(result.laplacian, jnp.asarray(-21.0, dtype=jnp.float32))

    def test_bound_linen_method_rule_does_not_warn(self):
        class SquareModule(nn.Module):
            scale: float = 2.0

            def square(self, x):
                return self.scale * x**2

            def square_rule(self, x):
                scale = self.scale
                return LapTuple(
                    scale * x.x**2,
                    scale * 2 * x.x * x.jacobian,
                    jnp.full_like(x.x, -11.0),
                )

        module = SquareModule()
        params = module.init(
            jax.random.PRNGKey(0),
            jnp.array([1.0], dtype=jnp.float32),
            method=module.square,
        )
        bound = module.bind(params)

        @custom_laplacian
        def square(x):
            return bound.square(x)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            square.def_laplacian_rule(bound.square_rule)

        assert not any(
            "closes over nontrivial Python state" in str(warning.message)
            for warning in caught
        )

        result = forward_laplacian(lambda value: jnp.sum(square(value)))(
            jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        )
        assert_allclose(result.laplacian, jnp.asarray(-33.0, dtype=jnp.float32))
