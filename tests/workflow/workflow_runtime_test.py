# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.runtime import DistributedConfig
from jaqmc.workflow.base import Workflow


def test_initialize_runtime_applies_settings_in_order(mocker):
    workflow = Workflow()
    calls: list[str] = []
    mocker.patch.object(
        workflow.logging, "apply", side_effect=lambda: calls.append("logging")
    )
    mocker.patch.object(workflow.jax, "apply", side_effect=lambda: calls.append("jax"))
    mocker.patch.object(
        workflow.distributed,
        "init_runtime",
        side_effect=lambda: calls.append("distributed"),
    )

    workflow.initialize_runtime()

    assert calls == ["logging", "jax", "distributed"]


def test_initialize_runtime_initializes_distributed(mocker):
    initialize = mocker.patch("jax.distributed.initialize")

    Workflow(
        distributed=DistributedConfig(
            coordinator_address="127.0.0.1:1234",
            num_processes=2,
            process_id=0,
        ),
    ).initialize_runtime()

    initialize.assert_called_once_with(
        "127.0.0.1:1234",
        num_processes=2,
        process_id=0,
        initialization_timeout=300,
    )


def test_initialize_runtime_skips_distributed_init_on_dry_run(mocker):
    initialize = mocker.patch("jax.distributed.initialize")

    Workflow(
        distributed=DistributedConfig(
            coordinator_address="127.0.0.1:1234",
            num_processes=2,
            process_id=0,
        ),
    ).initialize_runtime(dry_run=True)

    initialize.assert_not_called()
