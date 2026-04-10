# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import socket
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout

import pytest


class _Tee:
    """A simple Tee implementation to write to multiple files."""

    def __init__(self, *files):
        self.files = [f for f in files if f is not None]

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        # We don't want to close sys.stdout or sys.stderr,
        # but absl.logging might call close() on the stream.
        # Just flush everything.
        self.flush()


def redirect_stdout_stderr(f):
    """Redirect both stdout and stderr to a buffer f, while also printing to console.

    Args:
        f: A file-like object to capture the output.

    Returns:
        An ExitStack context manager that performs the redirection.
    """
    stack = ExitStack()
    stack.enter_context(redirect_stdout(_Tee(sys.stdout, f)))
    stack.enter_context(redirect_stderr(_Tee(sys.stderr, f)))
    return stack


def get_available_gpus():
    """Get list of available GPU device IDs.

    Returns:
        List of integer device IDs.
    """
    try:
        import jax

        gpu_devices = jax.devices("gpu")
        return list(range(len(gpu_devices)))
    except Exception:
        return []


def find_free_port():
    """Find a free port on localhost.

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_envs(num_processes, mode="cpu"):
    gpu_ids = get_available_gpus() if mode == "gpu" else []
    if mode == "gpu" and len(gpu_ids) < num_processes:
        pytest.skip(f"Need {num_processes} GPUs, but only {len(gpu_ids)} available")

    return [
        {"CUDA_VISIBLE_DEVICES": str(gpu_ids[process_id])}
        if mode == "gpu"
        else {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
        for process_id in range(num_processes)
    ]
