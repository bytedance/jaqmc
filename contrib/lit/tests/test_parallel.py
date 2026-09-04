# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="arg-type, assignment, misc, operator"

"""Parity tests for the single-frequency LIT data-parallel kernels."""

from functools import reduce
from operator import itemgetter

import jax
import numpy as np
import pytest
from jaqmc_contrib_lit.config import LITConfig
from jaqmc_contrib_lit.optimization import (
    _regularized_action_gradient,
    _solve_sr_direction_chunked,
    _solve_sr_direction_data_parallel,
)
from jaqmc_contrib_lit.pool import (
    _add_source_sums,
    _merge_source_sums_across_devices,
    _shard_batched_data_across_local_devices,
    _shard_rng_across_local_devices,
    _slice_batched_data,
)
from jaqmc_contrib_lit.response import LITSourceSums, WeightedComplexMoments
from jaqmc_contrib_lit.workflow import LITWorkflow
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.data import BatchedData
from jaqmc.utils import parallel_jax


def _replicated_specs(tree):
    return jax.tree.map(lambda _: parallel_jax.SHARE_PARTITION, tree)


def _data_specs(tree):
    return jax.tree.map(lambda _: parallel_jax.DATA_PARTITION, tree)


def _device_put_with_specs(values, specs):
    return jax.device_put(values, parallel_jax.make_sharding(specs))


def _assert_tree_allclose(actual, expected, *, rtol=2e-5, atol=2e-6):
    actual_leaves, actual_structure = jax.tree.flatten(actual)
    expected_leaves, expected_structure = jax.tree.flatten(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def _per_device_source_sums(device_count: int) -> LITSourceSums:
    index = jnp.arange(device_count, dtype=jnp.float32)
    # Deliberately give each shard a different local overflow-protection scale.
    ratio_scale = 10.0 ** jnp.linspace(-30.0, 30.0, device_count)
    ratio_sum = (1.0 + 0.25j) * (index + 1.0)
    ratio_abs2_sum = 0.5 + 0.2 * index
    psi_weight_sum = 0.8 + 0.3 * index
    psi_weight_sq_sum = 0.4 + 0.1 * index
    psi_log_ratio_abs2_sum = (-0.2 + 0.15 * index) * psi_weight_sum
    return LITSourceSums(
        sample_count=3.0 + index,
        weight_sum=2.0 + 0.5 * index,
        valid_sample_count=2.0 + index,
        ratio_scale=ratio_scale,
        ratio_sum=ratio_sum.astype(jnp.complex64),
        ratio_abs2_sum=ratio_abs2_sum,
        psi_weight_sum=psi_weight_sum,
        psi_weight_sq_sum=psi_weight_sq_sum,
        psi_log_ratio_abs2_sum=psi_log_ratio_abs2_sum,
        response_conj_over_source_sum=(0.2 - 0.3j) * (index + 1.0),
        ground_energy_sum=-1.0 - 0.2 * index,
        response_over_source_moments=WeightedComplexMoments(
            weight_sum=2.0 + 0.5 * index,
            origin=(0.7 - 0.2j) * (index + 1.0),
            mean_offset=jnp.zeros_like(index, dtype=jnp.complex64),
            centered_abs2_sum=0.3 + 0.1 * index,
        ),
        hbar_over_source_moments=WeightedComplexMoments(
            weight_sum=2.0 + 0.5 * index,
            origin=(-0.4 + 0.1j) * (index + 1.0),
            mean_offset=jnp.zeros_like(index, dtype=jnp.complex64),
            centered_abs2_sum=1.2 + 0.4 * index,
        ),
        psi_weight_max=0.25 + 0.05 * index,
    )


def test_merge_source_sums_matches_ordered_serial_merge():
    device_count = jax.local_device_count()
    per_device = _per_device_source_sums(device_count)

    def merge(local_arrays):
        local_sums = jax.tree.map(itemgetter(0), local_arrays)
        return _merge_source_sums_across_devices(local_sums)

    merge_sharded = parallel_jax.jit_sharded(
        merge,
        in_specs=(_data_specs(per_device),),
        out_specs=_replicated_specs(per_device),
    )
    (sharded_per_device,) = _device_put_with_specs(
        (per_device,),
        (_data_specs(per_device),),
    )
    parallel_result = merge_sharded(sharded_per_device)

    serial_shards = [
        jax.tree.map(lambda value, i=i: value[i], per_device)
        for i in range(device_count)
    ]
    serial_result = reduce(_add_source_sums, serial_shards)

    _assert_tree_allclose(parallel_result, serial_result)


def test_regularized_action_gradient_matches_serial_global_batch():
    device_count = jax.local_device_count()
    local_batch = 4
    batch_size = local_batch * device_count
    parameter_count = 5
    key_score, key_ratio, key_weight = jax.random.split(
        jax.random.PRNGKey(921),
        3,
    )
    score = jax.random.normal(key_score, (batch_size, parameter_count)) + 1j * (
        0.4
        * jax.random.normal(
            jax.random.fold_in(key_score, 1),
            (batch_size, parameter_count),
        )
    )
    ratio = jax.random.normal(key_ratio, (batch_size,)) + 1j * (
        0.3 * jax.random.normal(jax.random.fold_in(key_ratio, 1), (batch_size,))
    )
    source_weight = 0.2 + jax.random.uniform(key_weight, (batch_size,))
    # Exercise the global validity mask as well as the global max/sum reductions.
    ratio = ratio.at[local_batch].set(jnp.asarray(jnp.nan + 0.0j, ratio.dtype))
    source_weight = source_weight.at[-1].set(jnp.asarray(-1.0, source_weight.dtype))
    reverse_kl_weight = 0.17
    eps = 1e-7

    serial_result = _regularized_action_gradient(
        score,
        ratio,
        source_weight,
        reverse_kl_weight=reverse_kl_weight,
        eps=eps,
    )

    def distributed(local_score, local_ratio, local_source_weight):
        return _regularized_action_gradient(
            local_score,
            local_ratio,
            local_source_weight,
            reverse_kl_weight=reverse_kl_weight,
            eps=eps,
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )

    distributed_sharded = parallel_jax.jit_sharded(
        distributed,
        in_specs=(
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
        ),
    )
    sharded_args = _device_put_with_specs(
        (score, ratio, source_weight),
        (
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
    )
    parallel_result = distributed_sharded(*sharded_args)

    _assert_tree_allclose(parallel_result, serial_result, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("mode", ["off", "local_devices"])
def test_every_parallel_mode_rejects_multiple_jax_processes(monkeypatch, mode):
    workflow = object.__new__(LITWorkflow)
    workflow.lit_config = LITConfig()
    workflow.lit_config.parallel.mode = mode
    monkeypatch.setattr(jax, "process_count", lambda: 2)

    with pytest.raises(ValueError, match="exactly one JAX process"):
        workflow._validate_data_parallel_config()


def _parallel_sr_solve(score, grad, damping, *, kernel_null_vectors=None):
    device_count = jax.local_device_count()
    if kernel_null_vectors is None:

        def solve(local_score, replicated_grad, replicated_damping):
            return _solve_sr_direction_data_parallel(
                local_score,
                replicated_grad,
                replicated_damping,
                device_count=device_count,
            )

        in_specs = (
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
        )
        args = (score, grad, damping)
    else:

        def solve(
            local_score,
            replicated_grad,
            replicated_damping,
            local_null_vectors,
        ):
            return _solve_sr_direction_data_parallel(
                local_score,
                replicated_grad,
                replicated_damping,
                device_count=device_count,
                local_kernel_null_vectors=local_null_vectors,
            )

        in_specs = (
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
            PartitionSpec(None, parallel_jax.BATCH_AXIS_NAME),
        )
        args = (score, grad, damping, kernel_null_vectors)

    solve_sharded = parallel_jax.jit_sharded(
        solve,
        in_specs=in_specs,
        out_specs=parallel_jax.SHARE_PARTITION,
    )
    return solve_sharded(*_device_put_with_specs(args, in_specs))


def _serial_sr_solve(score, grad, damping, *, kernel_null_vectors=None):
    device_count = jax.local_device_count()
    chunk_rows = (score.shape[0] // device_count,) * device_count

    def score_chunk(index):
        row_count = chunk_rows[index]
        return score[index * row_count : (index + 1) * row_count]

    return _solve_sr_direction_chunked(
        chunk_rows,
        score_chunk,
        grad,
        damping,
        kernel_null_vectors=kernel_null_vectors,
    )


@pytest.mark.multi_device
@pytest.mark.skipif(jax.local_device_count() < 2, reason="needs multiple devices")
@pytest.mark.parametrize("solver_branch", ["primal", "dual"])
def test_multi_device_sr_matches_serial_solver(solver_branch):
    device_count = jax.local_device_count()
    sample_count = 3 * device_count
    parameter_count = 3 if solver_branch == "primal" else sample_count + 5
    score = jax.random.normal(
        jax.random.PRNGKey(31),
        (sample_count, parameter_count),
    )
    grad = jax.random.normal(jax.random.PRNGKey(32), (parameter_count,))
    damping = jnp.asarray(0.13, dtype=score.dtype)
    null_vectors = None
    if solver_branch == "dual":
        null_vectors = jnp.stack(
            (
                jnp.ones(sample_count, dtype=score.dtype),
                jnp.linspace(-1.0, 1.0, sample_count, dtype=score.dtype),
            )
        )

    parallel_result = _parallel_sr_solve(
        score,
        grad,
        damping,
        kernel_null_vectors=null_vectors,
    )
    serial_result = _serial_sr_solve(
        score,
        grad,
        damping,
        kernel_null_vectors=null_vectors,
    )

    np.testing.assert_allclose(parallel_result, serial_result, rtol=3e-5, atol=3e-6)


@pytest.mark.multi_device
@pytest.mark.skipif(jax.local_device_count() < 2, reason="needs multiple devices")
def test_multi_device_rngs_are_independent_and_walker_slices_stay_sharded():
    device_count = jax.local_device_count()
    rngs = _shard_rng_across_local_devices(jax.random.PRNGKey(913))
    keys = np.asarray(rngs).reshape(device_count, 2)
    assert rngs.sharding.spec == parallel_jax.DATA_PARTITION
    assert np.unique(keys, axis=0).shape[0] == device_count

    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.arange(
                24 * device_count,
                dtype=jnp.float32,
            ).reshape(8 * device_count, 1, 3),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.ones((1,), dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    sharded = _shard_batched_data_across_local_devices(batch)
    sliced = _slice_batched_data(
        sharded,
        2 * device_count,
        4 * device_count,
    )

    assert sliced.batch_size == 4 * device_count
    assert sliced.data.electrons.sharding.spec[0] == parallel_jax.BATCH_AXIS_NAME
    np.testing.assert_array_equal(
        sliced.data.electrons,
        batch.data.electrons[2 * device_count : 6 * device_count],
    )
