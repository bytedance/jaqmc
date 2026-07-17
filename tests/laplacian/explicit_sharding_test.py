# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JAX explicit sharding compatibility.

Requires JAX >= 0.7.0 for jax.set_mesh, jax.sharding.reshard, auto_axes, etc.
See https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html
"""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from jaqmc.laplacian import forward_laplacian
from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.sharding_fixtures import deterministic_two_layer_mlp

try:
    from jax.sharding import AxisType, auto_axes, reshard
except ImportError:
    pytest.skip(
        "Explicit sharding APIs (AxisType, auto_axes, reshard) are unavailable",
        allow_module_level=True,
    )


def _explicit_mesh(*shape_names):
    """Create an explicit-mode mesh.  shape_names = ((n, 'name'), ...)."""
    shape, names = zip(*shape_names)
    return jax.make_mesh(shape, names, axis_types=(AxisType.Explicit,) * len(shape))


# ---------------------------------------------------------------------------
# Basic explicit sharding
# ---------------------------------------------------------------------------


class TestExplicitSharding:
    """forward_laplacian under an explicit-sharding mesh."""

    def test_replicated_under_explicit_mesh(self):
        """Replicated (unsharded) input still works under an explicit mesh."""
        fn = lambda x: jnp.sum(jnp.sin(x))
        x = jnp.array([0.5, -0.3, 1.2])

        mesh = _explicit_mesh((1, "i"))
        with jax.set_mesh(mesh):
            result = forward_laplacian(fn)(x)
            check_with_brute_force(fn, x, actual_result=result)

    def test_resharded_input_sets_explicit_derivative_specs(self):
        """Auto-seeded derivatives keep explicit-sharding specs."""
        fn = lambda x: jnp.sum(jnp.tanh(x))
        x = jnp.array([0.5, -0.3, 1.2])

        mesh = _explicit_mesh((1, "i"))
        with jax.set_mesh(mesh):
            x_s = reshard(x, P("i"))
            assert x_s.sharding.mesh == mesh
            assert x_s.sharding.spec == P("i")

            result = forward_laplacian(fn)(x_s)
            assert result.x.sharding.mesh == mesh
            assert result.x.sharding.spec == P()
            assert result.dense_jacobian.sharding.mesh == mesh
            assert result.dense_jacobian.sharding.spec == P(None)
            assert result.laplacian.sharding.mesh == mesh
            assert result.laplacian.sharding.spec == P()
            check_with_brute_force(fn, x, actual_result=result)


# ---------------------------------------------------------------------------
# JIT + explicit sharding
# ---------------------------------------------------------------------------


class TestExplicitShardingJIT:
    """JIT-compiled forward_laplacian under explicit sharding."""

    def test_jit_mlp(self):
        mlp, x = deterministic_two_layer_mlp()

        mesh = _explicit_mesh((1, "i"))
        with jax.set_mesh(mesh):
            x_s = reshard(x, P("i"))
            result = jax.jit(forward_laplacian(mlp))(x_s)
            check_with_brute_force(mlp, x, actual_result=result, rtol=1e-4)


# ---------------------------------------------------------------------------
# auto_axes mode
# ---------------------------------------------------------------------------


class TestAutoAxes:
    """forward_laplacian under auto_axes sharding mode."""

    def test_auto_axes_mlp(self):
        mlp, x = deterministic_two_layer_mlp()

        mesh = jax.make_mesh((1,), ("i",), axis_types=(AxisType.Auto,))

        @auto_axes
        def run(x):
            return forward_laplacian(mlp)(x)

        with jax.set_mesh(mesh):
            result = run(x, out_sharding=P())
            check_with_brute_force(mlp, x, actual_result=result, rtol=1e-4)


class TestExplicitShardingMultiDevice:
    """Device-count-scaled explicit mesh with globally resharded input."""

    def test_resharded_input_matches_brute_force(self):
        if jax.device_count() < 2:
            pytest.skip("requires at least two devices")

        n_devices = jax.device_count()
        fn = lambda x: jnp.sum(jnp.tanh(x))
        x = jnp.arange(n_devices * 4, dtype=jnp.float32) / 10.0

        mesh = _explicit_mesh((n_devices, "i"))
        with jax.set_mesh(mesh):
            x_s = reshard(x, P("i"))
            result = forward_laplacian(fn)(x_s)
            check_with_brute_force(fn, x, actual_result=result)

    def test_dot_general_out_sharding_preserves_derivative_axis_layout(self):
        """A sharded dot output gains an unsharded leading derivative axis."""
        if jax.device_count() < 2:
            pytest.skip("requires at least two devices")

        n_devices = jax.device_count()
        mesh = _explicit_mesh((n_devices, "i"))
        lhs = (
            jnp.arange(n_devices * 6, dtype=jnp.float32).reshape(2 * n_devices, 3)
            / 10.0
        )
        rhs = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 20.0

        def dot(left, right):
            return jax.lax.dot_general(
                left,
                right,
                dimension_numbers=(((1,), (0,)), ((), ())),
                out_sharding=P("i", None),
            )

        def dot_ref(left, right):
            """Numerical reference without explicit output sharding."""
            return jax.lax.dot_general(
                left,
                right,
                dimension_numbers=(((1,), (0,)), ((), ())),
            )

        with jax.set_mesh(mesh):
            result = jax.jit(forward_laplacian(dot))(lhs, rhs)
            assert result.x.sharding.spec == P("i", None)
            assert result.dense_jacobian.sharding.spec == P(None, "i", None)
            assert result.laplacian.sharding.spec == P("i", None)
            # Explicit-sharding Hessian VJP cotangents can lose the output's
            # varying axis, so the oracle must not use explicit out_sharding.
            check_with_brute_force(
                dot, lhs, rhs, actual_result=result, expected_fn=dot_ref
            )
