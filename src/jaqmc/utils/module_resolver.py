# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import uuid
from importlib import import_module
from typing import Any


class ModuleResolutionError(ValueError):
    """Raised when a module reference cannot be resolved."""


def split_module_and_object(name: str) -> tuple[str, str | None]:
    """Parse ``module[:object]`` notation.

    Args:
        name: The module reference to parse.

    Returns:
        The module name and optional explicit object name.

    Raises:
        ModuleResolutionError: If the reference has empty segments or
            more than one colon.
    """
    if not name:
        raise ModuleResolutionError(
            "invalid module reference '': the module segment is empty. Expected "
            "`module` or `module:object`; `module` may also be a path to a .py file."
        )

    colon_count = name.count(":")
    if colon_count == 0:
        return name, None
    if colon_count > 1:
        reason = "a module reference can contain one colon"
    else:
        module, obj_name = name.split(":")
        if not module:
            reason = "the module segment before ':' is empty"
        elif not obj_name:
            reason = "the object segment after ':' is empty"
        else:
            return module, obj_name

    raise ModuleResolutionError(
        f"invalid module reference '{name}': {reason}. Expected `module` or "
        "`module:object`; `module` may also be a path to a .py file."
    )


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
        ModuleNotFoundError: Requested Python module or one of its dependencies
            cannot be imported.
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
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Clean staled temporary entries
            if sys.modules.get(module_id) is module:
                del sys.modules[module_id]
            raise
        return module
    if package:
        try:
            return import_module("." + module_name, package=package)
        except ModuleNotFoundError as e:
            relative_module = f"{package}.{module_name}"
            if (
                e.name != package
                and e.name != relative_module
                and not relative_module.startswith(f"{e.name}.")
            ):
                # Import error within target module
                raise
            return import_module(module_name)
    return import_module(module_name)


def resolve_object(name: str, package: str | None = None) -> Any:
    """Resolve object from ``module:object`` notation with default export support.

    The default object (without explicitly specified object name) is
    the primary object via ``__all__[0]`` in the target module.

    Args:
        name: The ``module:object`` notation. Supported forms:

            - ``"module:object"``: Explicitly resolve ``module.object``
              (e.g., ``"optax:adam"`` resolves to ``optax.adam``).
            - ``"module"``: Resolve default object from ``module.__all__[0]``
              (e.g., ``"jaqmc.optimizer.kfac"`` resolves to ``kfac``).
        package: Base package for relative imports. When specified, tries
            relative import first, then falls back to absolute import.
            If ``None``, only absolute imports are attempted.

    Returns:
        The resolved callable, class, or other object.

    Raises:
        ModuleResolutionError: The reference is malformed, cannot be imported,
            or cannot select an object from its imported module.

    Examples:
        Explicit ``module:object`` form:

        >>> resolve_object("optax:adam")
        <function adam at ...>

        Relative resolution via ``package``:

        >>> resolve_object("schedule:Standard", package="jaqmc.optimizer")
        <class 'jaqmc.optimizer.schedule.Standard'>
    """
    module, obj_name = split_module_and_object(name)
    try:
        module_obj = import_module_or_file(module, package)
    except ModuleNotFoundError as e:
        missing_module = e.name or str(e)
        if module == missing_module or module.startswith(f"{missing_module}."):
            detail = (
                f"could not import module '{module}': {e}. Check its spelling, "
                "whether it is installed in this environment, and the import path."
            )
        else:
            detail = (
                f"could not import module '{module}' because its dependency "
                f"'{missing_module}' is unavailable: {e}. Install that dependency "
                "in this environment."
            )
        raise ModuleResolutionError(
            f"module reference '{name}' is well-formed but {detail}"
        ) from e
    except OSError as e:
        raise ModuleResolutionError(
            f"could not load Python file '{module}': {e}. Check that the path exists "
            "and is readable."
        ) from e

    if obj_name is None:
        if not getattr(module_obj, "__all__", []):
            raise ModuleResolutionError(
                f"shorthand module reference '{name}' imported, but module '{module}' "
                "does not define a default object in __all__. Use an explicit "
                f"`{module}:object` reference instead."
            )
        obj_name = module_obj.__all__[0]
    try:
        obj = getattr(module_obj, obj_name)
    except AttributeError as e:
        raise ModuleResolutionError(
            f"module '{module}' imported, but does not export the requested object "
            f"'{obj_name}'. Check the object name in `{module}:object`."
        ) from e
    if obj is None:
        raise ModuleResolutionError(
            f"module reference '{name}' resolved to None: module '{module}' exports "
            f"object '{obj_name}' as None."
        )
    return obj
