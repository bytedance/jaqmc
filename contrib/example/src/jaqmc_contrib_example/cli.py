# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CLI for the example JaQMC contribution."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import click


@click.group(help="Example JaQMC contribution commands.")
def cli() -> None:
    pass


def _show_jaqmc_version() -> None:
    try:
        jaqmc_version = package_version("jaqmc")
    except PackageNotFoundError:
        jaqmc_version = "not installed"
    click.echo(f"jaqmc version: {jaqmc_version}")


@cli.command(help="Show JaQMC version information via the core package API.")
def version() -> None:
    _show_jaqmc_version()


@cli.command(help="Print a short description of this contribution.")
def info() -> None:
    from jaqmc.utils.config import ConfigManager

    config = ConfigManager({"package_name": "jaqmc-contrib-example"})
    click.echo("jaqmc-contrib-example: reference workspace contribution")
    click.echo("CLI entry point: example-contrib")
    click.echo(f"configured package: {config.get('package_name', '')}")
    _show_jaqmc_version()
