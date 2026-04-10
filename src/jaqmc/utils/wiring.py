# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
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
2. **From context**: Use ``wire(component, context)`` to inject from a dict.

Example:
    Programmatic use::

        estimator = KineticEnergy(f_log_psi=wf.evaluate)

    Context-driven use (e.g., from config engine)::

        estimator = KineticEnergy()  # only config fields set
        wire(estimator, f_log_psi=wf.evaluate)
"""

from dataclasses import field, fields, is_dataclass
from typing import Any

__all__ = ["check_wired", "runtime_dep", "wire"]

_UNSET = object()


class LateInit[ValueT]:
    """Descriptor for required runtime deps — raises on access before assignment."""

    def __set_name__(self, owner, name):
        self._name = f"_{name}"
        self._public_name = name

    def __get__(self, obj, tp=None) -> ValueT:
        if obj is None:
            return self  # type: ignore[return-value]
        if not hasattr(obj, self._name):
            cls_name = type(obj).__name__
            raise AttributeError(
                f"{cls_name}.{self._public_name} is a runtime dependency that was "
                f"not set. Wire it after construction: "
                f"`instance.{self._public_name} = ...` "
                f"or use wire(instance, {self._public_name}=...)"
            )
        return getattr(obj, self._name)

    def __set__(self, obj, val: ValueT):
        if not isinstance(val, LateInit):
            setattr(obj, self._name, val)

    def __repr__(self) -> str:
        return "LateInit()"


def runtime_dep(*, default: Any = _UNSET) -> Any:
    """Declare a runtime dependency field on a dataclass.

    Required deps (no default) use the LateInit descriptor, which raises
    a clear :class:`AttributeError` if accessed before being set. Optional deps
    use a regular field with the given default.

    Both kinds are invisible to serde and marked for :func:`wire`
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
        >>> est.f_eval  # raises before wiring
        Traceback (most recent call last):
            ...
        AttributeError: Estimator.f_eval is a runtime dependency ...
    """
    return field(
        default=default if default is not _UNSET else LateInit(),
        repr=False,
        metadata={"runtime": True, "serde_skip": True},
    )


def wire[ComponentT](component: ComponentT, **kwargs: Any) -> ComponentT:
    """Wire runtime dependencies into a dataclass component.

    Sets ``runtime_dep`` fields from context/kwargs and validates that all
    required deps are satisfied.

    Args:
        component: A dataclass instance with ``runtime_dep`` fields.
        **kwargs: Additional deps (override context).

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
        elif isinstance(f.default, LateInit):
            missing.append(f.name)
    if missing:
        raise ValueError(
            f"{type(component).__name__} requires runtime deps {missing} "
            f"but they were not provided. Available: {list(kwargs.keys())}"
        )

    return component


def check_wired(component: object, *, label: str = "") -> None:
    """Verify all required runtime deps on a dataclass have been set.

    Use this to catch missing wiring early — at configure time rather than
    mid-execution when a ``_LateInit`` descriptor would raise.

    Args:
        component: A dataclass instance to validate.
        label: Optional label for error context (e.g., estimator name).

    Raises:
        ValueError: If any required runtime deps are unset.
    """
    if not is_dataclass(component) or isinstance(component, type):
        return

    missing = []
    for f in fields(component):
        if not f.metadata.get("runtime"):
            continue
        if isinstance(f.default, LateInit) and not hasattr(component, f"_{f.name}"):
            missing.append(f.name)

    if missing:
        cls_name = type(component).__name__
        prefix = f"{label}: " if label else ""
        raise ValueError(
            f"{prefix}{cls_name} has unwired runtime deps: {missing}. "
            f"Wire them before passing to a stage, e.g.: "
            + ", ".join(f"instance.{m} = ..." for m in missing)
        )
