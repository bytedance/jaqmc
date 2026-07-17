# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shard_map compatibility."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from jaqmc.laplacian import (
    LapTuple,
    custom_laplacian,
    forward_laplacian,
    make_laplacian_input,
)
from jaqmc.utils import parallel_jax
from tests.laplacian.helpers import assert_allclose, brute_force_oracle

VMAPPED_LAPTUPLE_OUT_AXES = LapTuple.pytree_spec(0, 1, 0)
VMAPPED_LAPTUPLE_OUT_SPECS = LapTuple.pytree_spec(
    parallel_jax.DATA_PARTITION,
    jax.sharding.PartitionSpec(None, parallel_jax.BATCH_AXIS_NAME),
    parallel_jax.DATA_PARTITION,
)


@pytest.mark.skipif(
    (0, 5) <= jax.__version_info__ < (0, 7),
    reason="shard_map has pvary/tangent incompatibilities in JAX 0.5–0.6",
)
class TestShardMap:
    def test_custom_laplacian_loop_carry_marks_partition_varying_state(self):
        """Loop-carried LapTuple state keeps shard_map VMA metadata stable."""

        @custom_laplacian
        def score(x):
            return jnp.sum(x**2)

        @score.def_laplacian_rule
        def _(x):
            dense_jacobian = x.dense_jacobian
            return LapTuple(
                jnp.sum(x.x**2),
                jnp.sum(2 * x.x * dense_jacobian, axis=-1),
                jnp.full((), -321.0, dtype=x.x.dtype),
            )

        def run(x):
            logprob = jax.vmap(
                forward_laplacian(score),
                out_axes=VMAPPED_LAPTUPLE_OUT_AXES,
            )(x)
            logprob = LapTuple(
                parallel_jax.pvary(logprob.x),
                parallel_jax.pvary(logprob.jacobian),
                parallel_jax.pvary(logprob.laplacian),
            )

            def body(_, logprob):
                cond = parallel_jax.pvary(jnp.ones_like(logprob.x, dtype=bool))
                logprob_2 = jax.vmap(
                    forward_laplacian(score),
                    out_axes=VMAPPED_LAPTUPLE_OUT_AXES,
                )(x + 1.0)
                return LapTuple(
                    jnp.where(cond, logprob_2.x, logprob.x),
                    jnp.where(cond[None, ...], logprob_2.jacobian, logprob.jacobian),
                    jnp.where(cond, logprob_2.laplacian, logprob.laplacian),
                )

            return jax.lax.fori_loop(0, 1, body, logprob)

        compiled = parallel_jax.jit_sharded(
            run,
            in_specs=parallel_jax.DATA_PARTITION,
            out_specs=VMAPPED_LAPTUPLE_OUT_SPECS,
            check_vma=True,
        )

        n_devices = jax.device_count()
        x = jnp.arange(n_devices * 3, dtype=jnp.float32).reshape(n_devices, 3)
        x = jax.device_put(x, parallel_jax.make_sharding(parallel_jax.DATA_PARTITION))

        result = compiled(x)
        assert result.x.shape == (n_devices,)
        expected_x = jnp.sum((x + 1.0) ** 2, axis=-1)
        expected_jacobian = jnp.moveaxis(2 * (x + 1.0), 0, 1)
        assert_allclose(result.x, expected_x)
        assert result.jacobian.shape == expected_jacobian.shape
        assert_allclose(result.dense_jacobian, expected_jacobian)
        assert_allclose(result.laplacian, jnp.full((n_devices,), -321.0))

    def test_elementwise_nonlinear(self):
        """Elementwise ops (exp, sin) return full LapTuple through shard_map."""
        fn = lambda x: jnp.sum(jnp.exp(jnp.sin(x)))
        n_devices = jax.device_count()
        x_ref = (
            jnp.arange(n_devices * 3, dtype=jnp.float32).reshape(n_devices, 3) / 10.0
        )
        x = jax.device_put(
            x_ref, parallel_jax.make_sharding(parallel_jax.DATA_PARTITION)
        )
        mesh = parallel_jax.make_mesh()

        @partial(
            parallel_jax.shard_map,
            mesh=mesh,
            in_specs=(parallel_jax.DATA_PARTITION,),
            out_specs=VMAPPED_LAPTUPLE_OUT_SPECS,
        )
        @partial(jax.vmap, in_axes=(0,), out_axes=VMAPPED_LAPTUPLE_OUT_AXES)
        def run(x):
            return forward_laplacian(fn)(x)

        with jax.set_mesh(mesh):
            result = run(x)
        expected = jax.vmap(
            lambda x: brute_force_oracle(fn, x),
            out_axes=VMAPPED_LAPTUPLE_OUT_AXES,
        )(x_ref)
        assert_allclose(result.x, expected.x, rtol=1e-4)
        assert_allclose(result.dense_jacobian, expected.dense_jacobian, rtol=1e-4)
        assert_allclose(result.laplacian, expected.laplacian, rtol=1e-4)

    def test_sparse_square_jit_sharded(self):
        """Sparse-seeded square inside jit_sharded with check_vma=True."""
        fn = lambda x: jnp.sum(jnp.square(x))
        n_devices = jax.device_count()
        x_ref = (
            jnp.arange(n_devices * 6, dtype=jnp.float32).reshape(n_devices, 2, 3) / 10.0
        )
        x = jax.device_put(
            x_ref, parallel_jax.make_sharding(parallel_jax.DATA_PARTITION)
        )

        def evaluate(positions):
            def single(walker):
                seed = make_laplacian_input(walker, sparse_axis=0)
                return forward_laplacian(fn)(seed).laplacian

            return jax.vmap(single)(positions)

        compiled = parallel_jax.jit_sharded(
            evaluate,
            in_specs=parallel_jax.DATA_PARTITION,
            out_specs=parallel_jax.DATA_PARTITION,
            check_vma=True,
        )

        result = compiled(x)
        expected_laplacian = jax.vmap(
            lambda row: brute_force_oracle(fn, row).laplacian
        )(x_ref)
        assert_allclose(result, expected_laplacian, rtol=1e-4)
