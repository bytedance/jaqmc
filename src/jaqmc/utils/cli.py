# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence

import click

from jaqmc.utils import yaml_format


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
            workflow([load_yaml(f) for f in yml], dotlist, dry_run=dry_run)
        except ValueError as e:
            raise click.ClickException(str(e)) from None

    return command


def load_yaml(f) -> Mapping[str, object]:
    """Load YAML file and check if it's a dictionary.

    Args:
        f: File handler for the YAML file.

    Returns:
        Plain dictionary config.

    Raises:
        ValueError: YAML root is not a mapping.
    """
    source = getattr(f, "name", "<stream>")
    config = yaml_format.load_yaml(f)
    if not isinstance(config, dict):
        got = "empty document" if config is None else type(config).__name__
        raise ValueError(
            f"Invalid YAML config in '{source}': expected a mapping at the "
            f"document root, got {got}."
        )
    return config
