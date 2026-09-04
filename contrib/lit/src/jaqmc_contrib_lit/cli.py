# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry points for the optional LIT package."""

import click

from jaqmc.utils.cli import make_cli
from jaqmc.utils.config import ConfigManager


@click.group(help="Electric-dipole Lorentz integral transform workflows.")
def lit() -> None:
    """Run and postprocess LIT calculations."""


@lit.add_command
@make_cli(name="run", help="Compute a molecular electric-dipole LIT spectrum.")
def run_lit(cfg: ConfigManager, dry_run: bool) -> None:
    from jaqmc_contrib_lit.workflow import LITWorkflow

    LITWorkflow(cfg)(dry_run)


@lit.add_command
@make_cli(name="invert", help="Fit saved raw LIT archives on the CPU.")
def invert_lit(cfg: ConfigManager, dry_run: bool) -> None:
    from jaqmc_contrib_lit.postprocess import LITInversionPostprocessor

    LITInversionPostprocessor(cfg)(dry_run)
