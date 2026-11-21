# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry points for the optional LIT package."""

import click

from jaqmc.utils.cli import make_cli


@click.group(help="Electric-dipole Lorentz integral transform workflows.")
def lit() -> None:
    """Run and postprocess LIT calculations."""


@lit.add_command
@make_cli(name="run", help="Compute a molecular electric-dipole LIT spectrum.")
def run_lit(configs, dotlist, dry_run: bool) -> None:
    from jaqmc_contrib_lit.workflow import LITWorkflow

    workflow = LITWorkflow.from_config(configs, dotlist)
    workflow.initialize_runtime(dry_run)
    workflow(dry_run)


@lit.add_command
@make_cli(name="invert", help="Fit saved raw LIT archives on the CPU.")
def invert_lit(configs, dotlist, dry_run: bool) -> None:
    from jaqmc_contrib_lit.postprocess import LITInversionPostprocessor

    postprocessor = LITInversionPostprocessor.from_config(configs, dotlist)
    postprocessor.logging.apply()
    postprocessor(dry_run)
