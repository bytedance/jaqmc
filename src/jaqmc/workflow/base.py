# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar, Self

import jax
from upath import UPath

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager, configurable_dataclass
from jaqmc.utils.signal_handler import GracefulKiller

from .stage.base import RunContext

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "workflow"}
)


@configurable_dataclass
class ConfigCheck:
    """Controls config validation behavior.

    Args:
        ignore_extra: If True, silently ignore unrecognized config keys.
            If False, raise an error on extra keys.
        verbose: If True, print the fully resolved config with field
            descriptions at startup.
    """

    ignore_extra: bool = False
    verbose: bool = False


@configurable_dataclass
class WorkflowConfig:
    """Base configuration for workflows.

    Args:
        seed: Fixed random seed. If not provided, current time will be used.
        batch_size: Number of walkers (samples) to use in each iteration.
        save_path: Path to save checkpoints and logs. Can be any path
            supported by fsspec/universal_pathlib.
        restore_path: Path to restore checkpoints from. When set, checkpoints
            are restored from this path instead of ``save_path``. Can be a
            directory or a specific checkpoint file.
        config: Controls config validation behavior (extra-key warnings,
            verbose output).
        disable_jit: Disable JAX JIT compilation (for debugging).
    """

    seed: int | None = None
    batch_size: int = 4096
    save_path: str = ""
    restore_path: str = ""
    config: ConfigCheck = field(default_factory=ConfigCheck)
    disable_jit: bool = False

    def __post_init__(self):
        if not self.save_path:
            date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = f"jaqmc_{date_string}"
            project_root = _cwd_in_project_root()
            if project_root:
                self.save_path = str(project_root / "runs" / dirname)
            else:
                self.save_path = str(Path("runs") / dirname)
        if not self.restore_path:
            self.restore_path = self.save_path


class Workflow:
    """Base class for all workflows.

    Subclasses must override :meth:`run`.
    """

    cfg: ConfigManager

    config_class: ClassVar[type[WorkflowConfig]] = WorkflowConfig
    config: WorkflowConfig
    save_path: UPath
    restore_path: UPath
    signal_handler: GracefulKiller

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        """Return workflow-specific default config values.

        These presets are merged before user YAML and CLI overrides, so they act
        as documented workflow defaults rather than hard-coded overrides.
        """
        return {}

    def __init__(self, cfg: ConfigManager):
        cfg.use_preset(type(self).default_preset())
        self.cfg = cfg
        self.config = cfg.get("workflow", self.config_class)
        self.save_path = UPath(self.config.save_path)
        self.restore_path = (
            UPath(self.config.restore_path)
            if self.config.restore_path is not None
            else self.save_path
        )
        self.signal_handler = GracefulKiller()
        self.run_context = RunContext(
            save_path=self.save_path,
            restore_path=self.restore_path,
            signal_handler=self.signal_handler,
        )

    def run(self) -> None:
        """Execute the workflow.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def __call__(self, dry_run: bool = False) -> Self:
        self.prepare(dry_run=dry_run)
        if not dry_run:
            self.run()
        return self

    def prepare(self, dry_run: bool = False) -> None:
        """Finalize config and log startup info.

        On the master process, validates unused config keys and writes
        the resolved config to disk.
        """
        # Only write config and compare on the master process
        if jax.process_index() == 0:
            config_path = UPath(self.config.save_path) / "config.yaml"
            self.cfg.finalize(
                raise_on_unused=not self.config.config.ignore_extra,
                verbose=self.config.config.verbose,
                compare_yaml=config_path.read_text() if config_path.exists() else None,
            )
            if not dry_run:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(self.cfg.to_yaml())
        if dry_run:
            return
        logger.info(
            "Starting QMC with %i local XLA devices across %i processes.",
            jax.local_device_count(),
            jax.process_count(),
        )


def init_batched_data(
    data_init: Callable[[int, PRNGKey], Data | BatchedData],
    batch_size: int,
    rngs: PRNGKey,
    *,
    data_field: str = "electrons",
) -> BatchedData:
    """Create sharded and local batched data from a data initializer.

    Args:
        data_init: Function that creates initial data given a local
            batch size and a per-process random key.
        batch_size: Global batch size across all processes.
        rngs: Random key (will be folded per process).
        data_field: Name of the batched data field. Used when
            ``data_init`` returns a plain :class:`~jaqmc.data.Data` instance.

    Returns:
        Tuple of (sharded_batched_data, local_batched_data).

    Raises:
        ValueError: If batch size is not divisible by number of processes.
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
    sharded = jax.tree.map(
        jax.make_array_from_process_local_data,
        parallel_jax.make_sharding(local_data.partition_spec),
        local_data,
    )
    return sharded


def _cwd_in_project_root() -> Path | None:
    """Returns the project root by searching for pyproject.toml from CWD upwards."""
    path = Path.cwd()
    # Check current directory and its parents
    for p in [path, *path.parents]:
        pyproject = p / "pyproject.toml"
        if pyproject.is_file():
            try:
                # Check if it is the jaqmc project
                content = pyproject.read_text(encoding="utf-8")
                if 'name = "jaqmc"' in content or "name = 'jaqmc'" in content:
                    return p
            except OSError:
                pass
    return None
