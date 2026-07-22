# Copyright 2023 Microsoft Corporation
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2026.
#
# Original file was released under MIT, with the full license text
# available at licenses/folx_MIT.txt
#
# This file is distributed under the Apache License 2.0,
# with portions originally licensed under the MIT License.

"""Primitive registry state and built-in handler registration."""

import logging
from typing import Literal, overload

from jax.extend.core import Primitive

from ..types import LaplacianHandler

logger = logging.getLogger(__name__)

_REGISTRY: dict[Primitive | str, LaplacianHandler] = {}
_register_builtins_done_state = [False]


def _builtin_handler_tables() -> tuple[
    dict[Primitive | str, LaplacianHandler],
    ...,
]:
    from .arithmetic import ARITHMETIC_HANDLERS
    from .complex import COMPLEX_HANDLERS
    from .custom_laplacian import CUSTOM_LAPLACIAN_HANDLERS
    from .dot_general import DOT_GENERAL_HANDLERS
    from .dtype import DTYPE_HANDLERS
    from .elementwise import ELEMENTWISE_HANDLERS
    from .indexing import INDEXING_HANDLERS
    from .logical import LOGICAL_HANDLERS
    from .reductions import REDUCTIONS_HANDLERS
    from .selection import SELECTION_HANDLERS
    from .shape import SHAPE_HANDLERS
    from .slogdet import SLOGDET_HANDLERS
    from .stop_gradient import STOP_GRADIENT_HANDLERS

    return (
        ARITHMETIC_HANDLERS,
        COMPLEX_HANDLERS,
        CUSTOM_LAPLACIAN_HANDLERS,
        DTYPE_HANDLERS,
        DOT_GENERAL_HANDLERS,
        ELEMENTWISE_HANDLERS,
        INDEXING_HANDLERS,
        LOGICAL_HANDLERS,
        REDUCTIONS_HANDLERS,
        SELECTION_HANDLERS,
        SHAPE_HANDLERS,
        SLOGDET_HANDLERS,
        STOP_GRADIENT_HANDLERS,
    )


def register_builtin_handler(
    primitive_or_name: Primitive | str,
    handler: LaplacianHandler,
) -> None:
    _REGISTRY[primitive_or_name] = handler


def ensure_register_builtins() -> None:
    """Register all built-in primitive handlers atomically."""
    if _register_builtins_done_state[0]:
        return
    pending: dict[Primitive | str, LaplacianHandler] = {}
    for handler_table in _builtin_handler_tables():
        pending.update(handler_table)
    _REGISTRY.update(pending)
    _register_builtins_done_state[0] = True


@overload
def get_laplacian(
    primitive_or_name: Primitive | str, wrap_if_missing: Literal[True]
) -> LaplacianHandler: ...
@overload
def get_laplacian(
    primitive_or_name: Primitive | str, wrap_if_missing: bool = ...
) -> LaplacianHandler | None: ...
def get_laplacian(
    primitive_or_name: Primitive | str, wrap_if_missing: bool = False
) -> LaplacianHandler | None:
    """Return the registered handler, or fall back to ``wrap_general``.

    ``Primitive`` keys are the normal path for Jaxpr equation dispatch. String
    keys are reserved for named call sites such as ``pjit`` subexpressions and
    other synthetic handler entrypoints, so they do not support auto-wrapping.

    Raises:
        TypeError: If ``wrap_if_missing`` is ``True`` for a string name.
    """
    ensure_register_builtins()
    if primitive_or_name in _REGISTRY:
        return _REGISTRY[primitive_or_name]
    if wrap_if_missing:
        if isinstance(primitive_or_name, Primitive):
            from .core import wrap_general

            logger.warning(
                "%s not in registry. Full Hessian fallback will be slow.",
                primitive_or_name,
            )
            return wrap_general(primitive_or_name.bind)
        raise TypeError(f"Can't wrap {primitive_or_name} based on function names.")
    return None


def register_function(
    primitive_or_name: Primitive | str,
    handler: LaplacianHandler,
) -> None:
    """Register or replace a Forward Laplacian handler.

    Args:
        primitive_or_name: Registry key to install the handler under. Use a
            JAX ``Primitive`` for ordinary Jaxpr equation dispatch, or a
            string name for synthetic call sites such as named subexpressions.
        handler: Callable with the registry handler contract
            ``handler(args, kwargs)``. The handler receives the primitive call
            after tracing has packed positional arguments and keyword
            arguments, and should return either a plain primal result or a
            ``LapTuple``-wrapped result.

    Later registrations overwrite earlier ones for the same key. This is the
    normal entry point for custom rules and temporary handler replacement in
    tests.
    """
    ensure_register_builtins()
    register_builtin_handler(primitive_or_name, handler)


def deregister_function(primitive_or_name: Primitive | str) -> None:
    """Remove a previously registered Forward Laplacian handler.

    Args:
        primitive_or_name: Registry key to remove.

    Raises:
        KeyError: If no handler is currently registered for the given key.

    This removes both custom handlers and built-in handlers that have been
    registered into the live registry. Callers that temporarily deregister a
    built-in handler are responsible for restoring it explicitly.
    """
    ensure_register_builtins()
    if primitive_or_name not in _REGISTRY:
        raise KeyError(primitive_or_name)
    del _REGISTRY[primitive_or_name]
