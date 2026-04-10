# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Minimal Protocol-aware autodoc augmentation for Sphinx.

This extension supplements ``autoclass`` output for ``typing.Protocol`` classes:
- annotation-only protocol attributes (from ``__annotations__``).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, get_origin

from sphinx.application import Sphinx
from sphinx.util.typing import stringify_annotation


def _is_protocol_class(obj: Any) -> bool:
    return inspect.isclass(obj) and bool(getattr(obj, "_is_protocol", False))


def _iter_public_protocol_annotations(obj: Any) -> Iterable[tuple[str, Any]]:
    annotations = getattr(obj, "__annotations__", {})
    if not isinstance(annotations, dict):
        return []
    return [
        (name, annotation)
        for name, annotation in annotations.items()
        if not name.startswith("_")
    ]


def _type_target(annotation: Any) -> str:
    """Return a stable type target string for ``py:attribute :type:``."""
    base = get_origin(annotation) or annotation
    module = getattr(base, "__module__", None)
    qualname = getattr(base, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return stringify_annotation(base)


def _augment_protocol_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    del app, name, options
    if what != "class" or not _is_protocol_class(obj):
        return

    annotations = list(_iter_public_protocol_annotations(obj))
    if not annotations:
        return

    lines.append("")

    for member_name, annotation in annotations:
        lines.extend(
            [
                f".. py:attribute:: {member_name}",
                f"   :type: {_type_target(annotation)}",
                "",
            ]
        )


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("autodoc-process-docstring", _augment_protocol_docstring)
    return {"parallel_read_safe": True}
