# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import call

from jaqmc.utils.config import ConfigManager
from jaqmc.utils.runtime import JaxConfig, configure_runtime


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
