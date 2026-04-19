# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Startup-time runtime configuration for JaQMC applications."""

import logging
import sys
import time
from typing import Literal

from jaqmc.utils.config import ConfigManagerLike, configurable_dataclass

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "runtime"}
)


@configurable_dataclass
class LoggingConfig:
    """Configuration for JaQMC logging.

    Args:
        level: Minimum log level to emit. Messages below this severity are hidden.
        stream: Output stream for log messages.
    """

    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    stream: Literal["stdout", "stderr"] = "stderr"

    @property
    def stream_obj(self):
        if self.stream == "stdout":
            return sys.stdout
        elif self.stream == "stderr":
            return sys.stderr
        else:
            raise ValueError(f"Unknown stream type: {self.stream}")

    def apply(self):
        # JAX has it's own logging handlers and can cause repeated logs
        logging.getLogger("jax").handlers.clear()
        # Absl will pollute the root logger if there's no handler.
        if not logging.root.handlers:
            handler = logging.StreamHandler(self.stream_obj)
            formatter = CategoryFormatter(
                fmt="{levelname:.1} | {asctime} |{category:^10.10}| {message}",
                datefmt="%m-%d %H:%M:%S",
                style="{",
            )
            handler.setFormatter(formatter)
            handler.setLevel(self.level.upper())
            logging.root.addHandler(handler)
        logging.root.setLevel(self.level.upper())


class CategoryFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "category"):
            record.category = record.name.split(".", 1)[0]
        return super().format(record)


@configurable_dataclass
class JaxConfig:
    """Configuration for JAX global runtime flags.

    Args:
        enable_x64: Enable 64-bit types to be used.
            Note that neural networks parameters in Flax will stay float32 by default.
        debug_infs: Add inf checks to every operation.
            When an inf is detected on the output of a jit-compiled computation, call
            into the un-compiled version in an attempt to more precisely identify the
            operation which produced the inf.
        debug_nans: Add nan checks to every operation.
            When a nan is detected on the output of a jit-compiled computation, call
            into the un-compiled version in an attempt to more precisely identify the
            operation which produced the nan.
        default_matmul_precision: Control the default matmul precision for 32bit inputs.
            Leave unset to keep the JAX default.
        disable_jit: Disable JIT compilation and just call original Python.
    """

    enable_x64: bool = False
    debug_infs: bool = False
    debug_nans: bool = False
    disable_jit: bool = False
    default_matmul_precision: str = "float32"

    def apply(self) -> None:
        """Apply JAX global configuration flags."""
        import jax

        jax.config.update("jax_enable_x64", self.enable_x64)
        jax.config.update("jax_debug_infs", self.debug_infs)
        jax.config.update("jax_debug_nans", self.debug_nans)
        jax.config.update("jax_disable_jit", self.disable_jit)
        if self.default_matmul_precision is not None:
            jax.config.update(
                "jax_default_matmul_precision", self.default_matmul_precision
            )


@configurable_dataclass
class DistributedConfig:
    """Configuration for initializing JAX distributed runtime.

    Args:
        coordinator_address: IP address and port of the coordinator process
            (for example ``192.168.1.10:1234``).
        num_processes: Total number of processes in the distributed run.
        process_id: ID of the current process (``0`` to ``num_processes - 1``).
        initialization_timeout: Timeout in seconds for distributed runtime
            initialization.
        wait_second_before_connect: Seconds to wait before non-master processes
            connect to the coordinator.
    """

    coordinator_address: str | None = None
    num_processes: int = 1
    process_id: int = 0
    initialization_timeout: int = 300
    wait_second_before_connect: float = 10.0

    def init_runtime(self) -> None:
        """Initialize JAX distributed runtime for multi-host training."""
        import jax

        if self.coordinator_address is None or self.num_processes == 1:
            logger.info("Initialize a local runtime.")
            return

        logger.info("server_addr=%s", self.coordinator_address)
        logger.info("num_hosts=%s", self.num_processes)
        logger.info("host_idx=%s", self.process_id)

        if self.process_id > 0:
            logger.info(
                "Sleeping %s seconds before connecting to server...",
                self.wait_second_before_connect,
            )
            time.sleep(self.wait_second_before_connect)
        jax.distributed.initialize(
            self.coordinator_address,
            num_processes=self.num_processes,
            process_id=self.process_id,
            initialization_timeout=self.initialization_timeout,
        )


def configure_runtime(cfg: ConfigManagerLike, *, dry_run: bool = False) -> None:
    """Configure application startup state from config.

    Applies startup settings in this order:
    1. Logging
    2. JAX global configuration
    3. Distributed runtime initialization
    """
    cfg.get("logging", LoggingConfig).apply()
    cfg.get("jax", JaxConfig).apply()
    if not dry_run:
        cfg.get("distributed", DistributedConfig).init_runtime()
