# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Utilities for consistent YAML serialization."""

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import IO, Any

import yaml
from yaml.representer import SafeRepresenter


class ConfigLoader(yaml.SafeLoader):
    """Safe loader that also reads exponent floats such as ``1e-6``.

    YAML 1.1, which PyYAML implements, requires a decimal point and a signed
    exponent, so ``1e-6`` and ``1.0e6`` would otherwise load as strings.
    """


ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def load_yaml(stream: str | IO[str]) -> Any:
    """Load YAML with :class:`ConfigLoader`.

    Returns:
        The loaded Python value.
    """
    return yaml.load(stream, Loader=ConfigLoader)


def _is_numeric_list(lst: Iterable[Any]) -> bool:
    return bool(lst) and all(isinstance(item, (int, float)) for item in lst)


class CompactListDumper(yaml.SafeDumper):
    def represent_list(self, data: Iterable[Any]) -> yaml.SequenceNode:
        # Flat numeric list: [1, 2, 3]
        if _is_numeric_list(data):
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
        # Matrix (list of numeric lists): each row in flow style
        if data and all(
            isinstance(row, list) and _is_numeric_list(row) for row in data
        ):
            rows = [
                self.represent_sequence("tag:yaml.org,2002:seq", row, flow_style=True)
                for row in data
            ]
            return yaml.SequenceNode("tag:yaml.org,2002:seq", rows, flow_style=False)
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


CompactListDumper.add_representer(list, CompactListDumper.represent_list)
CompactListDumper.add_representer(tuple, CompactListDumper.represent_list)
# Enums are always subclasses: use `multi`
CompactListDumper.add_multi_representer(StrEnum, SafeRepresenter.represent_str)


def dump_yaml(data: Any, *, sort_keys: bool = False) -> str:
    r"""Dump data to YAML string with compact numeric lists.

    This function produces YAML output where lists containing only numbers
    are rendered in flow style (e.g., `[1, 2, 3]`) for better readability.

    Args:
        data: The data to dump. Can be a dict, list, or any YAML-serializable
            object.
        sort_keys: If True, dict keys are sorted alphabetically.

    Returns:
        A YAML string representation of the data.

    Examples:
        Numeric lists render in compact flow style:

        >>> print(dump_yaml({"lr": 0.05, "atoms": [1, 2, 3]}))
        lr: 0.05
        atoms: [1, 2, 3]
        <BLANKLINE>

        Matrices (lists of numeric lists) render each row in flow style:

        >>> print(dump_yaml({"matrix": [[1.0, 0.0], [0.0, 1.0]]}))
        matrix:
        - [1.0, 0.0]
        - [0.0, 1.0]
        <BLANKLINE>
    """
    return yaml.dump(
        data,
        Dumper=CompactListDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=sort_keys,
    )
