# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import textwrap
from unittest.mock import call

from jaqmc.utils.config import ConfigManager
from jaqmc.utils.runtime import JaxConfig, configure_runtime


def _run_cli_in_subprocess(
    cli_args: list[str], env_overrides: dict[str, str]
) -> dict[str, object]:
    script = textwrap.dedent(
        """
        import json
        import sys
        import warnings
        from click.testing import CliRunner
        from jaqmc.utils.cli import make_cli

        result_holder = {}

        @make_cli
        def check_workflow(cfg, dry_run=False):
            import jax
            from jax import numpy as jnp

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                arr = jnp.ones(1, dtype=jnp.float64)

            result_holder["dtype"] = str(arr.dtype)
            result_holder["warnings"] = [str(w.message) for w in caught]
            result_holder["jax_values"] = {
                k: jax.config.values[f"jax_{k}"]
                for k in (
                    "enable_x64",
                    "debug_infs",
                    "debug_nans",
                    "disable_jit",
                    "default_matmul_precision",
                )
            }
            result_holder["resolved_jax"] = cfg.resolved_config["jax"]

        res = CliRunner().invoke(check_workflow, sys.argv[1:])
        assert res.exit_code == 0, res.output
        print(json.dumps(result_holder))
        """
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        **env_overrides,
    }
    proc = subprocess.run(
        [sys.executable, "-c", script, *cli_args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_jax_config_apply_updates_enable_x64_and_matmul_precision(mocker):
    update = mocker.patch("jax.config.update")

    JaxConfig(enable_x64=True, default_matmul_precision="highest").apply()

    assert update.mock_calls == [
        call("jax_enable_x64", True),
        call("jax_debug_infs", False),
        call("jax_debug_nans", False),
        call("jax_disable_jit", False),
        call("jax_default_matmul_precision", "highest"),
    ]


def test_cli_startup_preserves_jax_env_vars_and_allows_cli_overrides():
    out = _run_cli_in_subprocess(
        cli_args=["jax.debug_infs=false"],
        env_overrides={
            "JAX_ENABLE_X64": "1",
            "JAX_DEBUG_INFS": "1",
            "JAX_DEBUG_NANS": "1",
            "JAX_DISABLE_JIT": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        },
    )

    expected = {
        "enable_x64": True,
        "debug_infs": False,
        "debug_nans": True,
        "disable_jit": True,
        "default_matmul_precision": "highest",
    }
    assert out["dtype"] == "float64"
    assert out["warnings"] == []
    assert out["jax_values"] == expected
    assert out["resolved_jax"] == expected


def test_cli_startup_enables_x64_from_config_without_truncation_warning():
    out = _run_cli_in_subprocess(
        cli_args=["jax.enable_x64=true"],
        env_overrides={"JAX_ENABLE_X64": "0"},
    )

    assert out["dtype"] == "float64"
    assert out["warnings"] == []
    assert out["jax_values"] == {
        "enable_x64": True,
        "debug_infs": False,
        "debug_nans": False,
        "disable_jit": False,
        "default_matmul_precision": "float32",
    }


def test_configure_runtime_applies_logging_jax_and_distributed_in_order(mocker):
    calls = []

    mocker.patch(
        "jaqmc.utils.runtime.LoggingConfig.apply",
        autospec=True,
        side_effect=lambda self: calls.append(f"logging:{self.level}"),
    )
    mocker.patch(
        "jaqmc.utils.runtime.JaxConfig.apply",
        autospec=True,
        side_effect=lambda _self: calls.append("jax"),
    )
    mocker.patch(
        "jaqmc.utils.runtime.DistributedConfig.init_runtime",
        autospec=True,
        side_effect=lambda _self: calls.append("distributed"),
    )

    cfg = ConfigManager(
        {
            "logging": {"level": "warning"},
            "jax": {
                "enable_x64": True,
                "default_matmul_precision": "highest",
            },
            "distributed": {
                "coordinator_address": "127.0.0.1:1234",
                "num_processes": 2,
                "process_id": 1,
            },
        }
    )

    configure_runtime(cfg)

    assert calls == ["logging:warning", "jax", "distributed"]


def test_configure_runtime_skips_distributed_init_on_dry_run(mocker):
    logging_apply = mocker.patch("jaqmc.utils.runtime.LoggingConfig.apply")
    jax_apply = mocker.patch("jaqmc.utils.runtime.JaxConfig.apply", autospec=True)
    init_runtime = mocker.patch(
        "jaqmc.utils.runtime.DistributedConfig.init_runtime", autospec=True
    )

    cfg = ConfigManager(
        {
            "logging": {"level": "info"},
            "jax": {"enable_x64": True},
            "distributed": {
                "coordinator_address": "127.0.0.1:1234",
                "num_processes": 2,
                "process_id": 1,
            },
        }
    )

    configure_runtime(cfg, dry_run=True)

    logging_apply.assert_called_once_with()
    jax_apply.assert_called_once()
    init_runtime.assert_not_called()
