# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import sys
import warnings

import chex
import pytest

multiprocessing.set_start_method("spawn", force=True)

PROXY_VARS = ["http_proxy", "https_proxy", "no_proxy"]


def pytest_addoption(parser):
    parser.addoption("--n-cpu-devices", type=int, default=2)
    parser.addoption("--keep-proxy-vars", action="store_true")


def pytest_configure(config):
    chex.set_n_cpu_devices(config.getoption("n_cpu_devices"))
    config.addinivalue_line("markers", "flaky: mark test as flaky (stochastic)")
    config.addinivalue_line(
        "markers", "integration: mark test as integration (runs full workflows)"
    )
    if not config.getoption("keep_proxy_vars") and any(
        os.environ.get(var) for var in PROXY_VARS
    ):
        warnings.warn(
            "Clearing proxy vars to avoid potential errors while testing distributed "
            "calculations. If they are actually needed, pass `--keep-proxy-vars` "
            "to keep them.",
            UserWarning,
        )
        for var in PROXY_VARS:
            if var in os.environ:
                del os.environ[var]


@pytest.fixture(autouse=True)
def cleanup_sys_modules():
    """Restore project modules after each test to prevent cross-test pollution.

    Only cleans up jaqmc.* modules to avoid interfering with stdlib or
    third-party modules that may maintain internal state (e.g., multiprocessing).
    """
    original_modules = {
        name: mod for name, mod in sys.modules.items() if name.startswith("jaqmc.")
    }
    yield

    for name in list(sys.modules.keys()):
        if not name.startswith("jaqmc."):
            continue
        if name not in original_modules:
            del sys.modules[name]
        elif sys.modules[name] is not original_modules[name]:
            sys.modules[name] = original_modules[name]
