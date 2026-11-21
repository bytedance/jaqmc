# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Generic runtime dependency wiring for dataclass components.

This module provides a unified pattern for declaring and injecting runtime
dependencies into dataclass-based components (estimators, optimizers, samplers,
writers, etc.).

Components have two kinds of fields:
- **Config fields**: User-configurable settings with defaults.
- **Runtime deps**: Dependencies injected at wiring time (``runtime_dep``).

Runtime deps can be set in two ways:
1. **Programmatically**: Pass directly in the constructor.
2. **After parsing**: Use ``wire(component, **deps)`` on a component built
   from configuration.

Example:
    Programmatic use::

        estimator = KineticEnergy(f_log_psi=wf.evaluate)

    Wiring a parsed component::

        estimator = KineticEnergy()  # only config fields set
        wire(estimator, f_log_psi=wf.evaluate)
"""

from dataclasses import field, fields, is_dataclass
from typing import Any

__all__ = ["UNSET", "check_wired", "runtime_dep", "wire"]


class _UnsetType:
    """Sentinel type for unset runtime dependencies."""

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET: Any = _UnsetType()


def runtime_dep(*, default: Any = UNSET) -> Any:
    """Declare a runtime dependency field on a dataclass.

    Both required (no default) and optional deps are excluded from
    configuration parsing and serialization, and marked for :func:`wire`
    discovery via ``metadata["runtime"]``.

    Args:
        default: Default value for optional runtime deps. Omit for required deps.

    Returns:
        A dataclass field descriptor.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Estimator:
        ...     mode: str = "fast"
        ...     f_eval: object = runtime_dep()
        >>> est = Estimator()
        >>> wire(est, f_eval=lambda x: x**2)  # doctest: +ELLIPSIS
        Estimator(...)
    """
    return field(
        default=default,
        repr=False,
        compare=False,
        kw_only=True,
        metadata={"runtime": True, "exclude": True, "validation_alias": ""},
    )


def wire[ComponentT](component: ComponentT, **kwargs: Any) -> ComponentT:
    """Wire runtime dependencies into a dataclass component.

    Sets ``runtime_dep`` fields whose names match keyword arguments, recursing
    into nested dataclass fields, and validates that all required deps are
    satisfied.

    Args:
        component: A dataclass instance with ``runtime_dep`` fields.
        **kwargs: Runtime deps keyed by field name.

    Returns:
        The wired component (for chaining).

    Type Parameters:
        ComponentT: Dataclass component type preserved as the return type.

    Raises:
        TypeError: If component is not a dataclass.
        ValueError: If required runtime deps are missing after wiring.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Estimator:
        ...     f_eval: object = runtime_dep()
        >>> est = Estimator()
        >>> wire(est, f_eval=lambda x: x**2)  # doctest: +ELLIPSIS
        Estimator(...)
        >>> est.f_eval(3)
        9
    """
    if not is_dataclass(component) or isinstance(component, type):
        raise TypeError(f"wire() expects a dataclass instance, got {type(component)}")

    missing = []
    for f in fields(component):
        if not f.metadata.get("runtime"):
            if is_dataclass(val := getattr(component, f.name)):
                wire(val, **kwargs)
            continue
        if f.name in kwargs:
            setattr(component, f.name, kwargs[f.name])
        elif getattr(component, f.name, UNSET) is UNSET:
            missing.append(f.name)
    if missing:
        raise ValueError(
            f"{type(component).__name__} requires runtime deps {missing} "
            f"but they were not provided. Available: {list(kwargs.keys())}"
        )

    return component


def check_wired(component: object, *, label: str = "") -> None:
    """Verify all required runtime deps on a dataclass have been set.

    Use this to catch missing wiring early at configure time rather than
    mid-execution.

    Args:
        component: A dataclass instance to validate.
        label: Optional label for error context (e.g., estimator name).

    Raises:
        ValueError: If any required runtime deps are unset.
    """
    if not is_dataclass(component) or isinstance(component, type):
        return

    missing = [
        f.name
        for f in fields(component)
        if f.metadata.get("runtime") and getattr(component, f.name, UNSET) is UNSET
    ]

    if missing:
        cls_name = type(component).__name__
        prefix = f"{label}: " if label else ""
        raise ValueError(
            f"{prefix}{cls_name} has unwired runtime deps: {missing}. "
            f"Wire them before passing to a stage, e.g.: "
            + ", ".join(f"instance.{m} = ..." for m in missing)
        )
