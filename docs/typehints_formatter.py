# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Picklable typehints formatter for sphinx-autodoc-typehints."""

from typing import TypeAliasType

import optax


class TypehintsFormatter:
    """Picklable callable so Sphinx can cache the config value."""

    def __call__(self, annotation, *args):
        if annotation == optax.ScalarOrSchedule:
            return ":py:obj:`optax.ScalarOrSchedule`"
        if annotation == optax.ScalarOrSchedule | None:
            return ":py:obj:`optax.ScalarOrSchedule` | :py:obj:`None`"
        if isinstance(annotation, TypeAliasType):
            return annotation.__name__
        return None

    def __reduce__(self):
        return (TypehintsFormatter, ())

    def __eq__(self, other):
        return isinstance(other, TypehintsFormatter)

    def __hash__(self):
        return hash(TypehintsFormatter)
