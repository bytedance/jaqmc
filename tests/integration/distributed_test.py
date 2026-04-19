# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Distributed multi-process training tests."""

import io
import multiprocessing
import operator
import os
from pathlib import Path

import h5py
import pytest

from ..utils.distributed import find_free_port, redirect_stdout_stderr, setup_envs


def init_runtime(cfg):
    """Set up logging and distributed runtime (mirrors make_cli behavior)."""
    from jaqmc.utils.runtime import configure_runtime

    configure_runtime(cfg)


def run_hydrogen_atom(distributed, save_path):
    from jaqmc.app.hydrogen_atom import hydrogen_atom_train_workflow
    from jaqmc.utils.config import ConfigManager

    cfg = ConfigManager(
        {
            "distributed": distributed,
            "workflow": {"save_path": save_path},
            "train": {"run": {"iterations": 1, "burn_in": 1}},
        }
    )
    init_runtime(cfg)
    hydrogen_atom_train_workflow(cfg)()


def run_molecule(distributed, save_path):
    from jaqmc.app.molecule import MoleculeTrainWorkflow
    from jaqmc.utils.config import ConfigManager

    cfg = ConfigManager(
        {
            "distributed": distributed,
            "system": {"module": "atom", "symbol": "H"},
            "wf": {"hidden_dims_single": [4, 4], "hidden_dims_double": [2, 2]},
            "workflow": {
                "save_path": save_path,
                "batch_size": 4,
            },
            "pretrain": {"run": {"iterations": 1, "burn_in": 1}},
            "train": {"run": {"iterations": 1, "burn_in": 1}},
        }
    )
    init_runtime(cfg)
    MoleculeTrainWorkflow(cfg)()


def run_solid(distributed, save_path):
    from jaqmc.app.solid import SolidTrainWorkflow
    from jaqmc.utils.config import ConfigManager

    cfg = ConfigManager(
        {
            "distributed": distributed,
            "system": {"module": "two_atom_chain", "vacuum_separation": 10.0},
            "wf": {"hidden_dims_single": [4, 4], "hidden_dims_double": [2, 2]},
            "workflow": {
                "save_path": save_path,
                "batch_size": 4,
            },
            "pretrain": {"run": {"iterations": 1, "burn_in": 1}},
            "train": {"run": {"iterations": 1, "burn_in": 1}},
        }
    )
    init_runtime(cfg)
    SolidTrainWorkflow(cfg)()


def worker_fn(run, distributed_config, save_path, queue, env_vars):
    """Worker function for multiprocessing distributed tests."""
    coordinator_address, num_processes, process_id = distributed_config
    for k, v in env_vars.items():
        os.environ[k] = v

    f = io.StringIO()
    with redirect_stdout_stderr(f):
        distributed = {
            "coordinator_address": coordinator_address,
            "num_processes": num_processes,
            "process_id": process_id,
            "initialization_timeout": 60,
            "wait_second_before_connect": 0.5,
        }
        try:
            run(distributed, save_path)
            exit_code = 0
        except Exception as e:
            import traceback

            print(f"Worker {process_id} failed: {e}")
            traceback.print_exc()
            exit_code = 1

    queue.put((process_id, exit_code, f.getvalue()))


def run_distributed_test(run, num_processes, tmp_path, mode="cpu"):
    """Run distributed test using multiprocessing.

    Returns:
        Sorted list of worker results.
    """
    coordinator_address = f"127.0.0.1:{find_free_port()}"
    queue = multiprocessing.Queue()

    processes = []
    for process_id, env_vars in enumerate(setup_envs(num_processes, mode)):
        distributed = (
            coordinator_address if num_processes > 1 else None,
            num_processes,
            process_id,
        )
        p = multiprocessing.Process(
            target=worker_fn,
            args=(run, distributed, str(tmp_path), queue, env_vars),
        )
        p.start()
        processes.append(p)

    try:
        results = [queue.get(timeout=180) for _ in range(num_processes)]
    except Exception:
        pytest.fail("Timed out waiting for worker results")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()

    return sorted(results, key=operator.itemgetter(0))


@pytest.mark.integration
@pytest.mark.parametrize(
    "run",
    [run_hydrogen_atom, run_molecule, run_solid],
)
@pytest.mark.parametrize(
    "num_processes, mode",
    [(2, "cpu"), (2, "gpu")],
)
def test_distributed_training(tmp_path: Path, run, num_processes, mode):
    """Unified test for distributed training across various configurations."""
    if num_processes != 1:
        # Old JAX version does not have good distributed running support on CPU
        pytest.importorskip("jax", minversion="0.6.0")

    results = run_distributed_test(
        run, num_processes=num_processes, tmp_path=tmp_path, mode=mode
    )

    for process_id, exit_code, output in results:
        if exit_code != 0:
            print(f"\n=== Process {process_id} output ===")
            print(output)
            pytest.fail(f"Process {process_id} failed with exit code {exit_code}")

    # Verification
    ckpts = list(tmp_path.glob("train_ckpt_*.npz"))
    assert ckpts, "No checkpoints were created"

    stats_h5 = tmp_path / "train_stats.h5"
    assert stats_h5.exists()
    with h5py.File(stats_h5, "r") as f:
        assert len(f["loss"]) == 1

    if num_processes > 1:
        found_log = any(
            "server_addr" in out or "num_hosts" in out for _, _, out in results
        )
        assert found_log, "No distributed initialization logs found"
    else:
        assert "2 local XLA devices across 1 processes" in results[0][2]
