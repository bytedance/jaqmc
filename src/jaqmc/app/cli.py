# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints for JaQMC applications.

Each application is a Click group with ``train`` and ``evaluate`` subcommands::

    jaqmc molecule train [OPTIONS] [DOTLIST]...
    jaqmc molecule evaluate [OPTIONS] [DOTLIST]...

To add a new application to the CLI:

1.  Define a new ``@cli.group()`` for the application.
2.  Add ``train`` and ``evaluate`` subcommands using ``@<group>.add_command``
    and ``@make_cli``. Import workflows lazily inside the command function.

Contributed applications register a ``click.Command`` or ``click.Group`` through
the ``jaqmc.apps`` entry-point group in a workspace package under ``contrib/``.
See ``contrib/README.md`` and ``docs/contrib/adding-contributions.md``.

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

from functools import cached_property
from importlib.metadata import EntryPoint, entry_points

import click

from jaqmc.utils.cli import make_cli
from jaqmc.utils.config import ConfigManager

ENTRY_POINT_GROUP = "jaqmc.apps"


class ContribAppsGroup(click.Group):
    """Click group that discovers contributed commands from entry points."""

    @cached_property
    def _contributed_entry_points(self) -> dict[str, EntryPoint]:
        """Discover and validate contributed command providers.

        Raises:
            click.ClickException: If providers use duplicate names or collide with
                built-in commands.
        """
        providers: dict[str, EntryPoint] = {}
        duplicate_names: set[str] = set()
        for entry_point in entry_points(group=ENTRY_POINT_GROUP):
            if entry_point.name in providers:
                duplicate_names.add(entry_point.name)
            providers[entry_point.name] = entry_point

        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise click.ClickException(f"duplicate jaqmc.apps providers: {names}")

        collision_names = sorted(set(self.commands) & set(providers))
        if collision_names:
            names = ", ".join(collision_names)
            raise click.ClickException(
                f"jaqmc.apps names collide with built-ins: {names}"
            )

        return providers

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List built-in and contributed command names.

        Returns:
            Sorted built-in and contributed command names.
        """
        builtin_names = super().list_commands(ctx)
        return sorted(set(builtin_names) | set(self._contributed_entry_points))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Resolve a built-in or load the selected contributed command.

        Returns:
            The resolved command, or ``None`` when no command matches.

        Raises:
            click.ClickException: If the selected provider is duplicated,
                collides with a built-in command, or does not return a Click
                command.
        """
        builtin = super().get_command(ctx, cmd_name)
        if builtin is not None:
            return builtin
        entry_point = self._contributed_entry_points.get(cmd_name)
        if entry_point is None:
            return None

        loaded = entry_point.load()
        if not isinstance(loaded, click.Command):
            raise click.ClickException(
                f"jaqmc.apps entry point '{cmd_name}' must return a "
                f"click.Command, got {type(loaded).__name__}"
            )
        return loaded


@click.group(
    cls=ContribAppsGroup,
    help="JaQMC: JAX-accelerated Quantum Monte Carlo framework.",
)
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


# --- electron_gas ---


@cli.group(name="electron-gas", help="Three-dimensional electron gas workflows.")
def electron_gas():
    pass


@electron_gas.add_command
@make_cli(name="train", help="Pretrain + train a homogeneous electron gas model.")
def electron_gas_train(cfg: ConfigManager, dry_run: bool):
    from .electron_gas import ElectronGasTrainWorkflow

    ElectronGasTrainWorkflow(cfg)(dry_run)


@electron_gas.add_command
@make_cli(name="evaluate", help="Evaluate a trained electron gas model.")
def electron_gas_evaluate(cfg: ConfigManager, dry_run: bool):
    from .electron_gas import ElectronGasEvalWorkflow

    ElectronGasEvalWorkflow(cfg)(dry_run)


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


# --- moire ---


@cli.group(help="Moire system workflows.")
def moire():
    pass


@moire.add_command
@make_cli(name="train", help="Train a moire system.")
def moire_train(cfg: ConfigManager, dry_run: bool):
    from .moire import MoireTrainWorkflow

    MoireTrainWorkflow(cfg)(dry_run)


@moire.add_command
@make_cli(name="evaluate", help="Evaluate a trained moire system.")
def moire_evaluate(cfg: ConfigManager, dry_run: bool):
    from .moire import MoireEvalWorkflow

    MoireEvalWorkflow(cfg)(dry_run)
