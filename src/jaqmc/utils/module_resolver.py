# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import uuid
from importlib import import_module
from typing import Any


def import_module_or_file(module_name: str, package: str | None = None) -> Any:
    """Import a python module or a python file.

    Args:
        module_name: The name of the module or file.
            If it ends with ".py", it will be considered as a file, otherwise module.
        package: The name of the base package to do relative imports.

    Returns:
        Contents of the module.

    Raises:
        OSError: Python file not found.
    """
    if module_name.endswith(".py"):
        # generate unique module name
        module_id = "jaqmc_" + str(uuid.uuid4()).replace("-", "_")
        # `imp` is deprecated. Using `importlib` way
        spec = importlib.util.spec_from_file_location(module_id, module_name)
        if spec is None or spec.loader is None:
            raise OSError(f"Failed to load {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_id] = module
        spec.loader.exec_module(module)
        return module
    if package:
        try:
            return import_module("." + module_name, package=package)
        except ModuleNotFoundError:
            return import_module(module_name)
    return import_module(module_name)


def resolve_object(name: str, package: str | None = None) -> Any:
    """Resolve object from ``module:name`` notation with default export support.

    The default object (without explicitly specified object name) is
    the primary object via ``__all__[0]`` in the target module.

    Args:
        name: The ``module:name`` notation. Supported forms:

            - ``"module:name"``: Explicitly resolve ``module.name``
              (e.g., ``"optax:adam"`` resolves to ``optax.adam``).
            - ``"module"``: Resolve default object from ``module.__all__[0]``
              (e.g., ``"jaqmc.optimizer.kfac"`` resolves to ``kfac``).
        package: Base package for relative imports. When specified, tries
            relative import first, then falls back to absolute import.
            If ``None``, only absolute imports are attempted.

    Returns:
        The resolved callable, class, or other object.

    Raises:
        ValueError: If the object cannot be resolved, including when using
            shorthand notation ("module") but the module has no `__all__`
            attribute defined.

    Examples:
        Explicit ``module:name`` form:

        >>> resolve_object("optax:adam")
        <function adam at ...>

        Relative resolution via ``package``:

        >>> resolve_object("schedule:Standard", package="jaqmc.optimizer")
        <class 'jaqmc.optimizer.schedule.Standard'>
    """
    colon_count = name.count(":")
    if colon_count == 0:
        module, obj_name = name, ""
    elif colon_count == 1:
        module, obj_name = name.split(":")
    else:
        raise ValueError(f"Invalid module '{name}': too many colons.")

    module_obj = import_module_or_file(module, package)
    if not obj_name:
        if not getattr(module_obj, "__all__", []):
            raise ValueError(f"Failed to find default object in {module}.")
        obj_name = module_obj.__all__[0]
    obj = getattr(module_obj, obj_name)
    if obj is None:
        raise ValueError(f"Fail to resolve '{name}'.")
    return obj
