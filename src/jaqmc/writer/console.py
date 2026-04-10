# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from jaqmc.utils.config import configurable_dataclass
from jaqmc.writer.base import Writer

__all__ = ["ConsoleWriter"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    """A single console output field.

    Args:
        key: Stat key to look up in the stats dictionary.
        alias: Display name. Defaults to ``key``.
        fmt: Python format specifier (e.g. ``".4f"``, ``"+.4f"``).
    """

    key: str
    alias: str | None = None
    fmt: str | None = None

    @property
    def display_name(self) -> str:
        return self.alias or self.key

    @staticmethod
    def parse(spec: str) -> "FieldSpec":
        """Parse ``[alias=]key[:format]``.

        Stat keys may contain colons (e.g. ``energy:kinetic``).  The last
        ``:``-separated segment is treated as a format specifier only when
        it contains a digit (e.g. ``.4f``, ``+.4f``).

        Returns:
            The parsed field specification.

        Examples:
            >>> FieldSpec.parse("total_energy:.6f")
            FieldSpec(key='total_energy', alias=None, fmt='.6f')

            >>> FieldSpec.parse("Lz=angular_momentum_z:+.4f")
            FieldSpec(key='angular_momentum_z', alias='Lz', fmt='+.4f')

            Colons in stat keys are preserved (only the last segment is
            checked for format specifiers):

            >>> FieldSpec.parse("energy:kinetic")
            FieldSpec(key='energy:kinetic', alias=None, fmt=None)
        """
        if "=" in spec:
            alias, rest = spec.split("=", 1)
        else:
            alias, rest = None, spec
        if ":" in rest:
            key, candidate = rest.rsplit(":", 1)
            if any(c.isdigit() for c in candidate):
                fmt = candidate
            else:
                key, fmt = rest, None
        else:
            key, fmt = rest, None
        return FieldSpec(key=key, alias=alias, fmt=fmt)


@configurable_dataclass
class ConsoleWriter(Writer):
    """Writes statistics to the console via the standard logger.

    Each field spec is ``[alias=]key[:format]``:

    - ``total_energy`` — display ``total_energy=...``
    - ``total_energy:.6f`` — with explicit format
    - ``energy=total_energy`` — display as ``energy=...``
    - ``Lz=angular_momentum_z:+.4f`` — alias + format

    Args:
        interval: Step interval for logging.
        fields: Comma-separated list of field specs.
    """

    interval: int = 1
    fields: str = "loss"

    def __post_init__(self):
        self.fields_specs = [FieldSpec.parse(f.strip()) for f in self.fields.split(",")]

    @contextmanager
    def open(self, working_dir, stage_name, initial_step: int = 0):
        self.logger = logging.LoggerAdapter(logger, extra={"category": stage_name})
        yield

    def write(self, step: int, stats: Mapping[str, Any]) -> None:
        if step % self.interval != 0:
            return

        values = {"step": step}
        for k, v in stats.items():
            values[k] = self.to_scalar(v)

        final_parts = [f"step={step}"]
        for spec in self.fields_specs:
            if spec.key in values:
                val = values[spec.key]
                final_parts.append(f"{spec.display_name}={val:{spec.fmt or '.4f'}}")
        self.logger.info(", ".join(final_parts))
