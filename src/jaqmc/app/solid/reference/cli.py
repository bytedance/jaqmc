# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for solid-state reference jobs."""

import logging
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import click
from upath import UPath

from jaqmc.app.solid.config import (
    SolidConfig,
    SolidPySCFSolverConfig,
    SolidQESolverConfig,
)
from jaqmc.utils.cli import load_yaml
from jaqmc.utils.config import ConfigError, ConfigManager
from jaqmc.utils.runtime import LoggingConfig

from . import pyscf, qe

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "reference"}
)


def _load_reference_inputs(
    yml: tuple[str, ...], dotlist: tuple[str, ...]
) -> tuple[SolidConfig, SolidPySCFSolverConfig | SolidQESolverConfig]:
    """Resolve the solid and solver configuration for a reference command.

    Returns:
        The configured solid and selected solver.
    """
    yaml_configs = []
    for path in yml:
        with open(path, encoding="utf-8") as stream:
            yaml_configs.append(load_yaml(stream))
    cfg = ConfigManager(yaml_configs, list(dotlist))
    cfg.get("logging", LoggingConfig).apply()
    system: SolidConfig | Callable[[], SolidConfig] = cfg.get_module(
        "system", SolidConfig
    )
    if callable(system):
        system = system()
    solver: SolidPySCFSolverConfig | SolidQESolverConfig = cfg.get_module(
        "solver", SolidPySCFSolverConfig
    )
    cfg.finalize()
    return system, solver


def _reference_job(
    system: SolidConfig,
    solver: SolidPySCFSolverConfig | SolidQESolverConfig,
) -> pyscf.SolidPySCFReferenceJob | qe.SolidQEReferenceJob:
    """Create the selected solver-specific reference job.

    Returns:
        A job bound to ``system`` and ``solver``.

    Raises:
        ConfigError: If the selected solver is unsupported.
    """
    if isinstance(solver, SolidQESolverConfig):
        return qe.SolidQEReferenceJob(system, solver)
    if isinstance(solver, SolidPySCFSolverConfig):
        return pyscf.SolidPySCFReferenceJob(system, solver)
    raise ConfigError(f"Unsupported solid solver config: {type(solver)!r}.")


@click.group(name="reference", help="Prepare and convert solid references.")
def reference() -> None:
    """Prepare solid-state orbital references."""


@reference.command(name="prepare", help="Write a solver-only solid reference job.")
@click.argument("dotlist", nargs=-1, required=False)
@click.option(
    "--yml",
    "--yaml",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path(s) to configuration YAML file(s). Multiple files are merged in order.",
)
@click.option("--output", required=True, type=click.Path(file_okay=False))
@click.option("--run", "run_job", is_flag=True, help="Run the generated solver job.")
@click.option(
    "--reference-output",
    type=click.Path(path_type=UPath),
    default=UPath("reference.npz"),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the resolved configuration and exit without writing a job.",
)
def prepare(
    dotlist: tuple[str, ...],
    yml: tuple[str, ...],
    output: str,
    run_job: bool,
    reference_output: UPath,
    dry_run: bool,
) -> None:
    """Write and optionally run a solid-state solver reference job.

    Raises:
        click.ClickException: If the configuration or solver job is invalid.
    """
    try:
        system, solver = _load_reference_inputs(yml, dotlist)
        job_dir = Path(output)
        target = UPath(reference_output)
        if not target.is_absolute():
            target = job_dir / target
        if dry_run:
            logger.info("Dry run: would write solver job to %s", job_dir)
            if run_job:
                logger.info("Dry run: would run the solver and write %s", target)
            else:
                logger.info(
                    "Dry run: rerun with --run to execute the solver and write %s",
                    target,
                )
            return

        job = _reference_job(system, solver)
        job.prepare(job_dir)

        logger.info("Wrote solver job to %s", job_dir)
        if not run_job:
            logger.info("Run `(cd %s && %s)`.", job_dir, shlex.join(job.command))
            logger.info(
                "Then `jaqmc solid reference convert --input %s --output %s`.",
                job_dir / job.output_path,
                target,
            )
            return

        logger.info("Running solver in %s.", job_dir)
        job.run(job_dir)
        job.finalize_npz(job_dir, target)
        logger.info("Wrote reference to %s", target)
    except ConfigError as e:
        raise click.ClickException(str(e)) from None
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from None


@reference.command(
    name="convert", help="Convert a PySCF or Quantum ESPRESSO output to reference.npz."
)
@click.option(
    "--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--output", "output_path", required=True, type=click.Path(path_type=UPath)
)
@click.option("--source", "source", type=click.Choice(["pyscf", "qe"]), default=None)
def convert(
    input_path: Path,
    output_path: UPath,
    source: Literal["pyscf", "qe"] | None = None,
) -> None:
    """Convert a solid solver output to a reference file.

    Raises:
        click.ClickException: If the solver output cannot be converted.
    """
    LoggingConfig().apply()
    try:
        if source is None:
            if input_path.is_file():
                source = "pyscf"
            elif (input_path / "data-file-schema.xml").is_file():
                source = "qe"
            else:
                raise click.ClickException(
                    f"Could not determine the solid reference source for {input_path}. "
                    "Expected a PySCF checkpoint file or a Quantum ESPRESSO "
                    "prefix.save directory."
                )
        if source == "qe":
            qe.convert_save_directory(input_path, output_path)
        elif source == "pyscf":
            pyscf.convert_checkpoint(input_path, output_path)
        else:
            raise click.ClickException(f"Invalid source {source} for conversion.")
        logger.info("Wrote reference to %s", output_path)
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from None
