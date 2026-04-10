# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Utilities for YAML processing and annotation."""

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

import yaml
from yaml.representer import SafeRepresenter


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


def annotate_yaml_with_sources(
    yaml_str: str,
    source_map: dict[str, tuple[str, int, str | None]],
    verbose: bool = False,
) -> str:
    """Annotate YAML string with source location comments.

    This function post-processes a YAML string to inject comments indicating
    where specific configuration sections (dataclasses/callables) are defined.
    It uses indentation to reconstruct the nested key structure (e.g., `a.b.c`)
    and looks up source locations in the provided source map.

    Args:
        yaml_str: The original YAML string to annotate.
        source_map: A dictionary mapping configuration paths to their source
            information. Key is the path (e.g., "system.molecule"), value is
            a tuple of (source_file, line_number, docstring).
        verbose: If True, also includes docstrings of the configuration objects
            in the comments.

    Returns:
        Annotated YAML string with definition comments.
    """
    if not source_map:
        return yaml_str

    lines = yaml_str.splitlines()
    result = []
    # Stack tracks (indentation_level, key_name) to reconstruct full path
    path_stack: list[tuple[int, str]] = []

    # Regex to match keys in YAML
    key_pattern = re.compile(r"^(\s*)([^:\s#]+)\s*:")

    for line in lines:
        match = key_pattern.match(line)
        if match:
            indent_str, key = match.groups()
            indent = len(indent_str)

            # Pop keys from stack that are deeper or same level
            while path_stack and path_stack[-1][0] >= indent:
                path_stack.pop()
            path_stack.append((indent, key))

            current_path = ".".join(k for _, k in path_stack)

            if current_path in source_map:
                src_file, lineno, doc = source_map[current_path]
                prefix = indent_str
                result.append(f"{prefix}# Defined in {src_file}:{lineno}")
                if verbose and doc:
                    for doc_line in doc.splitlines():
                        result.append(f"{prefix}# {doc_line}")

        result.append(line)
    return "\n".join(result)
