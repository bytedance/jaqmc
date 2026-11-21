# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Workflow configuration roots and common runtime lifecycle."""

import datetime
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar, Self

import jax
from pydantic import Field
from upath import UPath

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import (
    config_to_yaml,
    configurable_dataclass,
    parse_config,
)
from jaqmc.utils.runtime import DistributedConfig, JaxConfig, LoggingConfig
from jaqmc.utils.signal_handler import GracefulKiller

from .stage.base import RunContext

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "workflow"}
)


@configurable_dataclass
class WorkflowConfig:
    """Configuration shared by all workflows.

    Args:
        seed: Fixed random seed. If not provided, current time will be used.
        batch_size: Number of walkers (samples) to use in each iteration.
        save_path: Path to save checkpoints and logs. Can be any path
            supported by fsspec/universal_pathlib. If empty, defaults to a
            timestamped directory under ``runs/``.
        restore_path: Path to restore checkpoints from. If ``None``,
            ``save_path`` is used. Can be a directory or a specific checkpoint
            file.
    """

    seed: int | None = None
    batch_size: int = 4096
    save_path: str | None = None
    restore_path: str | None = None

    def __post_init__(self) -> None:
        if not self.save_path:
            date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = f"jaqmc_{date_string}"
            project_root = _cwd_in_project_root()
            self.save_path = str(
                (project_root / "runs" / dirname)
                if project_root
                else Path("runs") / dirname
            )


@configurable_dataclass
class Workflow:
    """Base class for configuration-driven workflows."""

    workflow_name: ClassVar[str | None] = None
    config: WorkflowConfig = Field(default_factory=WorkflowConfig, alias="workflow")
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    jax: JaxConfig = field(default_factory=JaxConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    @classmethod
    def from_config(
        cls,
        configs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        dotlist: Sequence[str] = (),
    ) -> Self:
        """Create a workflow from layered configuration mappings.

        Returns:
            An instance of the concrete workflow class.
        """
        return parse_config(cls, configs, dotlist)

    def to_yaml(self) -> str:
        """Serialize the public workflow configuration.

        Returns:
            Canonical YAML for this workflow.
        """
        return config_to_yaml(self)

    def initialize_runtime(self, dry_run: bool = False) -> None:
        """Apply process-wide runtime settings.

        Applies logging, then JAX global flags, then distributed runtime
        initialization. Distributed initialization is skipped when ``dry_run``
        is true.

        Args:
            dry_run: If true, apply logging and JAX flags but skip distributed
                runtime initialization.
        """
        self.logging.apply()
        self.jax.apply()
        if not dry_run:
            self.distributed.init_runtime()

    @property
    def save_path(self) -> UPath:
        """Return the canonical directory for checkpoints and logs.

        Returns:
            The absolute save directory.

        Raises:
            RuntimeError: Configuration initialization left ``save_path`` empty.
        """
        save_path = self.config.save_path
        if not save_path:
            raise RuntimeError("Workflow save_path is unset after initialization")
        return UPath(save_path).resolve()

    @property
    def restore_path(self) -> UPath:
        """Return the canonical restore location, defaulting to the save path.

        Both paths are resolved because :meth:`_validate_paths` compares them,
        and a relative path would never compare equal to its absolute form.

        Returns:
            The absolute restore directory or checkpoint file.
        """
        if not self.config.restore_path:
            return self.save_path
        return UPath(self.config.restore_path).resolve()

    @property
    def restore_dir(self) -> UPath:
        """Return the directory containing restore artifacts."""
        restore_path = self.restore_path
        return restore_path.parent if restore_path.is_file() else restore_path

    def _make_run_context(self) -> RunContext:
        save_path = self.save_path
        restore_path = self.restore_path
        return RunContext(
            save_path=save_path,
            restore_path=restore_path,
            signal_handler=GracefulKiller(),
        )

    def __call__(self, dry_run: bool = False) -> Self:
        self.prepare(dry_run=dry_run)
        if not dry_run:
            self.build_runtime()
            self.run()
        return self

    def build_runtime(self) -> None:
        """Construct transient runtime dependencies from the parsed workflow."""
        raise NotImplementedError

    def run(self) -> None:
        """Execute the workflow."""
        raise NotImplementedError

    def config_path(self) -> UPath:
        """Return the resolved configuration snapshot path.

        Returns:
            Path of the YAML snapshot for this workflow.
        """
        filename = (
            f"{self.workflow_name}_config.yaml" if self.workflow_name else "config.yaml"
        )
        return self.save_path / filename

    def config_backup_path(self) -> UPath:
        """Return an unused backup path for the current snapshot."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = self.config_path().stem
        history_dir = self.save_path / "config_history"
        backup_stem = f"{config_stem}.backup_{timestamp}"
        backup_path = history_dir / f"{backup_stem}.yaml"
        suffix = 1
        while backup_path.exists():
            backup_path = history_dir / f"{backup_stem}_{suffix}.yaml"
            suffix += 1
        return backup_path

    def prepare(self, dry_run: bool = False) -> None:
        """Log and optionally snapshot the resolved configuration.

        Writes the YAML snapshot under ``save_path`` unless ``dry_run`` is true.
        """
        if jax.process_index() == 0:
            self._validate_paths()
            config_path = self.config_path()
            previous_yaml = config_path.read_text() if config_path.exists() else None
            current_yaml = self.to_yaml()
            logger.info("Resolved configurations:\n%s", current_yaml)
            if not dry_run:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                if previous_yaml is not None and previous_yaml != current_yaml:
                    backup_path = self.config_backup_path()
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.rename(backup_path)
                config_path.write_text(current_yaml)
        if not dry_run:
            logger.info(
                "Starting QMC with %i local XLA devices across %i processes.",
                jax.local_device_count(),
                jax.process_count(),
            )

    def _validate_paths(self) -> None:
        """Reject a restore path that does not exist or an unusable ``save_path``.

        ``restore_path`` must exists unless it's the same as ``save_path`` (the case of
        fresh runs). If ``save_path`` differs from the restore directory, it should be
        empty, otherwise the files there can be overwritten and cause data loss.

        Raises:
            FileNotFoundError: Restore path does not exist.
            ValueError: Save path is not a directory, or is not empty
                during cross-directory resume.
        """
        save_path = self.save_path
        restore_path = self.restore_path
        if restore_path != save_path and not restore_path.exists():
            raise FileNotFoundError(f"Restore path {restore_path} does not exist")
        if save_path.exists() and not save_path.is_dir():
            raise ValueError(f"Save path is not a directory: {save_path}")

        restore_dir = self.restore_dir
        if save_path != restore_dir and save_path.exists() and any(save_path.iterdir()):
            raise ValueError(
                f"Save path is not empty during cross-directory resume: {save_path}"
            )


def init_batched_data(
    data_init: Callable[[int, PRNGKey], Data | BatchedData],
    batch_size: int,
    rngs: PRNGKey,
    *,
    data_field: str = "electrons",
) -> BatchedData:
    """Create process-sharded batched data from an initializer.

    Returns:
        The sharded batched data.

    Raises:
        ValueError: If the batch size is not process divisible.
    """
    rngs = jax.random.fold_in(rngs, jax.process_index())
    if batch_size % jax.process_count() != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by "
            f"number of processes {jax.process_count()}."
        )
    local_batch = batch_size // jax.process_count()
    raw = data_init(local_batch, rngs)
    local_data = raw if isinstance(raw, BatchedData) else BatchedData(raw, [data_field])
    local_data.check()
    return jax.tree.map(
        jax.make_array_from_process_local_data,
        parallel_jax.make_sharding(local_data.partition_spec),
        local_data,
    )


def _cwd_in_project_root() -> Path | None:
    """Return the JaQMC project root when running inside the repository."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            content = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "jaqmc"' in content or "name = 'jaqmc'" in content:
            return candidate
    return None
