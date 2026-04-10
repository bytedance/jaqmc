# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import click
import yaml

from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.logging_setup import LoggingLevel, setup_logging


def make_cli(workflow=None, **kwargs):
    if workflow is None:
        return lambda workflow: make_cli(workflow, **kwargs)

    default_name = workflow.__name__

    @click.command(**{"name": kwargs.pop("name", default_name), **kwargs})
    @click.argument("dotlist", nargs=-1, required=False)
    @click.option(
        "--yml",
        "--yaml",
        help="Path(s) to configuration YAML file(s). "
        "Multiple files are merged in order.",
        type=click.File(),
        multiple=True,
    )
    @click.option(
        "--dry-run",
        help="Print the resolved configuration and exit without running the workflow.",
        is_flag=True,
    )
    def command(
        dotlist: tuple[str, ...], yml: Sequence[str], dry_run: bool = False
    ) -> None:
        cfg = ConfigManager([load_yaml(f) for f in yml], list(dotlist))
        setup_logging(cfg.get("logging_level", LoggingLevel.info))
        if not dry_run:
            distributed_config = cfg.get("distributed", parallel_jax.DistributedConfig)
            distributed_config.init_runtime()
        workflow(cfg, dry_run=dry_run)

    return command


def load_yaml(f):
    """Load YAML file and check if it's a dictionary.

    Args:
        f: File handler for the YAML file.

    Returns:
        Plain dictionary config.

    Raises:
        ValueError: YAML contains list config.
    """
    config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"JaQMC expects a dictionary config from {f.read()}. Got {type(config)}."
        )
    return config
