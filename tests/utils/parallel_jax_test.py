# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for parallel JAX utilities, especially all_gather.

These tests verify that the all_gather function correctly gathers distributed
arrays from all devices and materializes them on each device.

Test Coverage:
- Single-process multi-device tests (simulated with XLA_FLAGS)
- Multi-process distributed tests (true distributed arrays)
- Comparison with process_allgather (JAX >= 0.8.X)
"""

import io
import multiprocessing
import operator
import os

import numpy as np
import pytest

from jaqmc.utils.runtime import DistributedConfig

from .distributed import find_free_port, redirect_stdout_stderr, setup_envs

# ============================================================================
# Single-process multi-device tests
# ============================================================================


def test_all_gather_single_process():
    """Test all_gather in single-process multi-device setting.

    Uses XLA_FLAGS to simulate multiple devices on a single machine.
    This tests the basic all_gather functionality without true distribution,
    including various shapes, dtypes, and pytrees.
    """
    import jax
    from jax import numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from jaqmc.utils.parallel_jax import BATCH_AXIS_NAME, all_gather, make_mesh

    if jax.device_count() < 2:
        pytest.skip("More than one devices are required to test gathering")
    mesh = make_mesh()

    def shard_array(arr):
        """Helper to shard an array along the batch axis.

        Returns:
            Sharded array.
        """
        sharding = NamedSharding(mesh, PartitionSpec(BATCH_AXIS_NAME))
        return jax.device_put(arr, sharding)

    # Test 1: Various array shapes and dtypes
    test_cases = [
        jnp.arange(20, dtype=jnp.float32).reshape(10, 2),  # 2D array
        jnp.ones((4, 3, 2), dtype=jnp.float32),  # 3D array
        jnp.arange(8, dtype=jnp.int32),  # 1D integer array
        jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),  # 2D float32
    ]

    for test_array in test_cases:
        sharded = shard_array(test_array)
        result = all_gather(sharded)
        assert result.shape == test_array.shape
        assert jnp.allclose(test_array, result, rtol=1e-6)

    # Test 2: Pytrees - exact data.py use case
    pytree_test = {
        "positions": jnp.arange(12, dtype=jnp.float32).reshape(6, 2),
        "momenta": jnp.ones((4, 3), dtype=jnp.float32),
        "nested": {
            "field_b": jnp.array([[1.0, 2.0]], dtype=jnp.float32).reshape(2, 1),
        },
    }

    # Shard all arrays in the pytree
    sharded_pytree = jax.tree.map(shard_array, pytree_test)

    # Apply all_gather to the whole pytree
    result_pytree = jax.tree.map(all_gather, sharded_pytree)

    # Compare all leaves in the pytree
    leaves_orig, _ = jax.tree.flatten(pytree_test)
    leaves_res, _ = jax.tree.flatten(result_pytree)

    for leaf_orig, leaf_res in zip(leaves_orig, leaves_res):
        assert leaf_res.shape == leaf_orig.shape
        assert jnp.allclose(leaf_orig, leaf_res, rtol=1e-6)


# ============================================================================
# Multi-process distributed tests (true distributed arrays)
# ============================================================================


def parallel_jax_worker(
    process_id, num_processes, coordinator_address, queue, env_vars, test_arrays
):
    """Worker function for distributed all_gather tests.

    Args:
        process_id: Process ID.
        num_processes: Total number of processes.
        coordinator_address: Coordinator address.
        queue: multiprocessing.Queue to send results back.
        env_vars: Dictionary of environment variables to set.
        test_arrays: List of arrays to test all_gather on.

    Raises:
        AssertionError: If sharding is not working as expected.
    """
    for k, v in env_vars.items():
        os.environ[k] = v

    f = io.StringIO()
    with redirect_stdout_stderr(f):
        try:
            import jax
            from jax.sharding import NamedSharding, PartitionSpec

            from jaqmc.utils.parallel_jax import BATCH_AXIS_NAME, all_gather, make_mesh

            # Initialize distributed runtime
            DistributedConfig(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=process_id,
                initialization_timeout=60,
                wait_second_before_connect=0.5,
            ).init_runtime()

            assert jax.process_index() == process_id
            mesh = make_mesh()
            sharding = NamedSharding(mesh, PartitionSpec(BATCH_AXIS_NAME))

            gathered_results = []
            for full_data in test_arrays:
                sharded_data = jax.device_put(full_data, sharding)
                if num_processes > 1:
                    try:
                        _ = np.array(sharded_data)
                        raise AssertionError("Sharding not working!")
                    except RuntimeError:
                        pass
                gathered_results.append(np.array(all_gather(sharded_data)))

            exit_code = 0
        except Exception:
            import traceback

            traceback.print_exc()
            exit_code = 1
            gathered_results = []

    queue.put((process_id, exit_code, f.getvalue(), gathered_results))


def run_parallel_jax_test(num_processes, test_arrays, mode="cpu"):
    """Run distributed all_gather test using multiprocessing.

    Returns:
        Sorted list of worker results.
    """
    coordinator_address = f"127.0.0.1:{find_free_port()}"
    queue = multiprocessing.Queue()

    processes = []
    for process_id, env_vars in enumerate(setup_envs(num_processes, mode)):
        p = multiprocessing.Process(
            target=parallel_jax_worker,
            args=(
                process_id,
                num_processes,
                coordinator_address,
                queue,
                env_vars,
                test_arrays,
            ),
        )
        p.start()
        processes.append(p)

    try:
        results = [queue.get(timeout=30) for _ in range(num_processes)]
    except Exception:
        pytest.fail("Timed out waiting for worker results")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()

    return sorted(results, key=operator.itemgetter(0))


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_all_gather_distributed_two_processes(mode: str):
    """Test all_gather in a true multi-process distributed setting with 2 processes."""
    pytest.importorskip("jax", minversion="0.5.0")

    # Verify gathered data
    test_arrays = [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.arange(4, dtype=np.int32),
    ]

    results = run_parallel_jax_test(num_processes=2, test_arrays=test_arrays, mode=mode)

    # Verify exit codes and output
    for process_id, exit_code, output, _ in results:
        if exit_code != 0:
            print(f"\n=== Process {process_id} output ===")
            print(output)
            pytest.fail(f"Process {process_id} failed with exit code {exit_code}")

    for process_id, _, _, gathered_list in results:
        assert len(gathered_list) == len(test_arrays)
        for gathered, expected in zip(gathered_list, test_arrays):
            assert gathered.shape == expected.shape
            assert np.allclose(gathered, expected, rtol=1e-6)
