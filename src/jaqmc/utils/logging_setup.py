# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from enum import StrEnum


class LoggingLevel(StrEnum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


class CategoryFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "category"):
            record.category = record.name.split(".", 1)[0]
        return super().format(record)


def setup_logging(log_level: LoggingLevel = LoggingLevel.info):
    # JAX has it's own logging handlers and can cause repeated logs
    logging.getLogger("jax").handlers.clear()
    # Absl will pollute the root logger if there's no handler.
    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = CategoryFormatter(
            fmt="{levelname:.1} | {asctime} |{category:^10.10}| {message}",
            datefmt="%m-%d %H:%M:%S",
            style="{",
        )
        handler.setFormatter(formatter)
        handler.setLevel(log_level.value.upper())
        logging.root.addHandler(handler)
    logging.root.setLevel(log_level.value.upper())
