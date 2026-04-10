# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints for JaQMC applications.

Each application is a Click group with ``train`` and ``evaluate`` subcommands::

    jaqmc molecule train [OPTIONS] [DOTLIST]...
    jaqmc molecule evaluate [OPTIONS] [DOTLIST]...

To add a new application to the CLI:

1.  Define a new ``@cli.group()`` for the application.
2.  Add ``train`` and ``evaluate`` subcommands using ``@<group>.add_command``
    and ``@make_cli``. Import workflows lazily inside the command function.

Example::

    @cli.group(help="Demo workflows.")
    def my_demo():
        pass

    @my_demo.add_command
    @make_cli(name="train", help="Train the demo.")
    def my_demo_train(cfg: ConfigManager, dry_run: bool):
        from jaqmc.app.my_demo import my_demo_train_workflow

        my_demo_train_workflow(cfg)(dry_run)
"""

import click

from jaqmc.utils.cli import make_cli
from jaqmc.utils.config import ConfigManager


@click.group(help="JaQMC: JAX-accelerated Quantum Monte Carlo framework.")
def cli():
    pass


# --- hydrogen_atom ---


@cli.group(help="Hydrogen atom demonstration workflows.")
def hydrogen_atom():
    pass


@hydrogen_atom.add_command
@make_cli(name="train", help="Train the hydrogen atom model.")
def hydrogen_atom_train(cfg: ConfigManager, dry_run: bool):
    from .hydrogen_atom import hydrogen_atom_train_workflow

    hydrogen_atom_train_workflow(cfg)(dry_run)


# --- molecule ---


@cli.group(help="Molecular system workflows.")
def molecule():
    pass


@molecule.add_command
@make_cli(name="train", help="Pretrain + train a molecular system.")
def molecule_train(cfg: ConfigManager, dry_run: bool):
    from .molecule import MoleculeTrainWorkflow

    MoleculeTrainWorkflow(cfg)(dry_run)


@molecule.add_command
@make_cli(name="evaluate", help="Evaluate a trained molecular system.")
def molecule_evaluate(cfg: ConfigManager, dry_run: bool):
    from .molecule import MoleculeEvalWorkflow

    MoleculeEvalWorkflow(cfg)(dry_run)


# --- solid ---


@cli.group(help="Solid-state system workflows.")
def solid():
    pass


@solid.add_command
@make_cli(name="train", help="Pretrain + train a solid-state system.")
def solid_train(cfg: ConfigManager, dry_run: bool):
    from .solid import SolidTrainWorkflow

    SolidTrainWorkflow(cfg)(dry_run)


@solid.add_command
@make_cli(name="evaluate", help="Evaluate a trained solid-state system.")
def solid_evaluate(cfg: ConfigManager, dry_run: bool):
    from .solid import SolidEvalWorkflow

    SolidEvalWorkflow(cfg)(dry_run)


# --- hall ---


@cli.group(help="Quantum Hall effect workflows.")
def hall():
    pass


@hall.add_command
@make_cli(name="train", help="Train a quantum Hall system.")
def hall_train(cfg: ConfigManager, dry_run: bool):
    from .hall import HallTrainWorkflow

    HallTrainWorkflow(cfg)(dry_run)


@hall.add_command
@make_cli(name="evaluate", help="Evaluate a trained quantum Hall system.")
def hall_evaluate(cfg: ConfigManager, dry_run: bool):
    from .hall import HallEvalWorkflow

    HallEvalWorkflow(cfg)(dry_run)
