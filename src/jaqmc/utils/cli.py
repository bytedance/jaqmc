# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import click
import yaml

from jaqmc.utils.config import ConfigError, ConfigManager
from jaqmc.utils.runtime import configure_runtime


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
        try:
            cfg = ConfigManager([load_yaml(f) for f in yml], list(dotlist))
            configure_runtime(cfg, dry_run=dry_run)
            workflow(cfg, dry_run=dry_run)
        except ConfigError as e:
            raise click.ClickException(str(e)) from None

    return command


def load_yaml(f):
    """Load YAML file and check if it's a dictionary.

    Args:
        f: File handler for the YAML file.

    Returns:
        Plain dictionary config.

    Raises:
        ConfigError: YAML root is not a mapping.
    """
    source = getattr(f, "name", "<stream>")
    config = yaml.safe_load(f)
    if not isinstance(config, dict):
        got = "empty document" if config is None else type(config).__name__
        raise ConfigError(
            f"Invalid YAML config in '{source}': expected a mapping at the "
            f"document root, got {got}."
        )
    return config
