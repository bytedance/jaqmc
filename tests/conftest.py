# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import sys
import warnings

import chex
import jax
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
    config.addinivalue_line(
        "markers",
        "x64_modes: run the test with JAX x64 disabled and enabled (scoped)",
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


def _import_enable_x64():
    if jax.__version_info__ >= (0, 8, 0):
        try:
            from jax import enable_x64
        except ImportError:
            from jax.experimental import enable_x64  # type: ignore[no-redef]
        return enable_x64
    try:
        from jax.experimental import enable_x64  # type: ignore[no-redef]
    except ImportError:
        return None
    return enable_x64


def pytest_generate_tests(metafunc):
    if metafunc.definition.get_closest_marker("x64_modes") is not None:
        metafunc.parametrize(
            "x64_mode",
            [False, True],
            ids=["x64_off", "x64_on"],
            indirect=True,
        )


@pytest.fixture(autouse=True)
def x64_mode(request):
    """Scope JAX x64 mode for tests marked with ``x64_modes``.

    Yields:
        ``None`` for unmarked tests, otherwise ``False`` (x64 off) or ``True`` (x64 on).
    """
    mode = getattr(request, "param", None)
    if mode is None:
        yield None
        return

    enable_x64 = _import_enable_x64()
    if mode:
        if enable_x64 is None:
            pytest.skip("requires JAX >= 0.8.0 for enable_x64")
        with enable_x64(True):
            assert jax.config.read("jax_enable_x64")
            yield True
    else:
        if enable_x64 is None:
            if jax.config.read("jax_enable_x64"):
                pytest.skip("cannot scope-disable x64 on JAX < 0.8.0")
            yield False
            return
        with enable_x64(False):
            assert not jax.config.read("jax_enable_x64")
            yield False


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
