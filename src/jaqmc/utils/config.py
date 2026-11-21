# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from functools import wraps
from types import UnionType
from typing import Any, Union, dataclass_transform, get_args, get_origin, overload

from pydantic import (
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
)
from pydantic.fields import FieldInfo
from pydantic_core import ArgsKwargs, CoreSchema, core_schema

from jaqmc.utils.module_resolver import ModuleResolutionError, resolve_object
from jaqmc.utils.yaml_format import dump_yaml, load_yaml

__all__ = [
    "ModuleConfig",
    "config_to_dict",
    "config_to_yaml",
    "configurable_dataclass",
    "is_configurable_dataclass",
    "materialize_config_default",
    "parse_config",
]

_materializing_schema_defaults: ContextVar[bool] = ContextVar(
    "materializing_schema_defaults", default=False
)


# =============================================================================
# Configurable dataclasses
# =============================================================================


@overload
def configurable_dataclass[T: type[Any]](cls: T, *, kw_only: bool = True) -> T: ...
@overload
def configurable_dataclass[T: type[Any]](
    cls: None = None, *, kw_only: bool = True
) -> Callable[[T], T]: ...
@dataclass_transform(kw_only_default=True, field_specifiers=(field, Field))
def configurable_dataclass(cls: Any = None, *, kw_only: bool = True) -> Any:
    """Decorator for dataclasses that participate in the config system.

    Applies ``@dataclass`` (if needed) and builds a Pydantic schema adapter
    from the class's resolved runtime field annotations (do not use
    ``from __future__ import annotations``). A class-defined ``__post_init__``
    runs after construction from the merged configuration.

    Returns:
        The decorated class.
    """

    def wrap(target: type[Any]) -> type[Any]:
        post_init = getattr(target, "__post_init__", None)
        if (
            post_init is not None
            and post_init.__qualname__ != "Module.__post_init__"
            and not hasattr(post_init, "__wrapped__")
        ):

            @wraps(post_init)
            def guarded_post_init(self: Any, *args: Any, **kwargs: Any) -> None:
                if _materializing_schema_defaults.get():
                    return
                post_init(self, *args, **kwargs)

            target.__post_init__ = guarded_post_init
        if "__dataclass_fields__" not in target.__dict__:
            target = dataclass(target, kw_only=kw_only)
        return _attach_config_schema(target)

    return wrap if cls is None else wrap(cls)


def _attach_config_schema(cls: type[Any]) -> type[Any]:
    for f in fields(cls):
        if not f.metadata.get("exclude") and "module" in (
            f.name,
            f.default.alias if isinstance(f.default, FieldInfo) else None,
        ):
            raise TypeError(f"{cls.__name__}: 'module' is a reserved config key.")

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        __tracebackhide__ = True
        cls.__jaqmc_adapter__.validator.validate_python(
            ArgsKwargs(args, kwargs), self_instance=self, by_name=True
        )

    cls.__pydantic_config__ = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=False
    )
    cls.__jaqmc_adapter__ = TypeAdapter(cls)
    cls.__init__ = __init__
    return cls


def is_configurable_dataclass(target: Any) -> bool:
    """Return whether ``target`` is a configurable dataclass type or instance."""
    cls = target if isinstance(target, type) else type(target)
    return is_dataclass(cls) and "__jaqmc_adapter__" in cls.__dict__


def materialize_config_default(schema: Callable[[], Any]) -> Any:
    """Return a value from a default config without running ``__post_init__``."""
    token = _materializing_schema_defaults.set(True)
    try:
        return schema()
    finally:
        _materializing_schema_defaults.reset(token)


# =============================================================================
# Polymorphic module configuration
# =============================================================================


@dataclass(frozen=True)
class ModuleConfig:
    """Pydantic annotation for polymorphic module configuration fields.

    Use with :data:`typing.Annotated` on fields of a
    :func:`configurable_dataclass`::

        optim: Annotated[
            OptimizerLike, ModuleConfig(package="jaqmc.optimizer")
        ] = field(default_factory=KFACOptimizer)

    The annotated field type supplies both the runtime module contract (any
    class or :func:`typing.runtime_checkable` protocol in the type) and any
    direct scalar value types (such as ``float`` or ``Literal[...]``).

    Args:
        package: Base package for resolving relative module paths.
            When omitted, only absolute module paths are accepted.
    """

    package: str | None = None

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_before_validator_function(
            self._deserialize,
            handler(source_type),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                self._serialize, info_arg=False
            ),
        )

    def _resolve_module(self, module_name: Any) -> type[Any]:
        if not isinstance(module_name, str):
            raise ValueError(
                f"Invalid config at 'module': expected a string, "
                f"got {type(module_name).__name__}."
            )
        try:
            target = resolve_object(module_name, package=self.package)
        except ModuleResolutionError as exc:
            raise ValueError(f"Invalid config at 'module': {exc}") from exc
        if not (isinstance(target, type) and is_configurable_dataclass(target)):
            raise ValueError(f"{target!r} is not a configurable dataclass.")
        return target

    def _deserialize(self, raw: Any) -> Any:
        if not isinstance(raw, Mapping):
            return raw
        if "module" not in raw:
            raise ValueError(
                "Missing required config key 'module': this field has no "
                "default module, so a mapping must select one."
            )
        layers = _module_layers(raw)
        target = self._resolve_module(layers[-1][0])
        # layers[0] holds the workflow's materialized default for this field
        # (including customizations from a parent's default_factory). If it
        # targets the same class, use it as the baseline; switching to another
        # module in a later layer resets earlier overrides back to this default.
        default = {
            **config_to_dict(materialize_config_default(target)),
            **(layers[0][1] if self._resolve_module(layers[0][0]) is target else {}),
        }
        data = default
        for name, arguments in layers[1:]:
            data = (
                _deep_merge(data, arguments, target)
                if self._resolve_module(name) is target
                else default
            )
        return _from_dict(target, data)

    @staticmethod
    def _serialize(value: Any, nxt: SerializerFunctionWrapHandler) -> Any:
        if not is_configurable_dataclass(value):
            return nxt(value)
        module_path = f"{type(value).__module__}:{type(value).__name__}"
        return {"module": module_path, **config_to_dict(value)}


# =============================================================================
# Parsing and serialization
# =============================================================================


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a configurable dataclass instance to a plain dictionary.

    Args:
        config: Configurable dataclass instance.

    Returns:
        Plain dictionary representation with aliases applied.

    Raises:
        ValueError: If ``config`` is not a configurable dataclass instance.
    """
    cls = type(config)
    if not is_configurable_dataclass(cls):
        raise ValueError(
            f"Could not serialize configuration: "
            f"{cls!r} is not a configurable dataclass."
        )
    return cls.__jaqmc_adapter__.dump_python(config, by_alias=True)


def config_to_yaml(config: Any) -> str:
    """Serialize a typed configuration to canonical plain YAML.

    Args:
        config: Configurable dataclass instance.

    Returns:
        YAML containing the public serializable configuration fields.
    """
    return dump_yaml(config_to_dict(config))


def parse_config[T](
    schema: type[T],
    configs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    dotlist: Sequence[str] = (),
) -> T:
    """Parse layered configuration into a fixed dataclass schema.

    Layers are merged in order and command-line overrides have the highest
    precedence.

    Args:
        schema: Default-constructible root configurable dataclass type.
        configs: One mapping or an ordered sequence of mappings.
        dotlist: ``key=value`` command-line overrides.

    Returns:
        An instance of ``schema``.

    Raises:
        ValueError: If a layer is not a mapping.
    """
    layers = [configs] if isinstance(configs, Mapping) else list(configs)
    data = config_to_dict(materialize_config_default(schema))
    for layer in (*layers, _dotlist_to_dict(list(dotlist))):
        if not isinstance(layer, Mapping):
            raise ValueError("Configuration layers must be mappings.")
        data = _deep_merge(data, layer, schema)
    return _from_dict(schema, data)


def _from_dict[T](schema: type[T], data: Mapping[str, Any]) -> T:
    if not (isinstance(schema, type) and is_configurable_dataclass(schema)):
        raise ValueError(f"{schema!r} is not a configurable dataclass.")
    adapter: TypeAdapter[T] = getattr(schema, "__jaqmc_adapter__")
    return adapter.validate_python(dict(data))


def _is_valid_config(cls: type[Any], data: Mapping[str, Any]) -> bool:
    try:
        materialize_config_default(lambda: _from_dict(cls, data))
        return True
    except (ValidationError, TypeError, ValueError):
        return False


def _module_layers(node: Mapping[str, Any]) -> tuple[tuple[Any, dict[str, Any]], ...]:
    """Return the ordered layers of a module node, oldest first."""
    if "module" not in node:
        return ()
    arguments = copy.deepcopy(dict(node))
    module = arguments.pop("module")
    return module if isinstance(module, tuple) else ((module, arguments),)


def _deep_merge(base: Any, override: Any, tp: Any = None) -> Any:
    """Recursively merge mappings without mutating either input.

    Returns:
        A deep copy of the merged mappings, or the overriding value.
    """
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        return copy.deepcopy(override)

    # Module names resolve only at deserialization, against the field's import
    # base, so merging keeps every layer's spelling in order.
    if "module" in base or "module" in override:
        layers = _module_layers(base)
        if "module" in override:
            layers = (*layers, *_module_layers(override))
        else:
            layers = (*layers, (layers[-1][0], copy.deepcopy(dict(override))))
        return {"module": layers}

    if get_origin(tp) in (Union, UnionType):
        variants = [
            a
            for a in get_args(tp)
            if isinstance(a, type) and is_configurable_dataclass(a)
        ]
        if len(variants) == 1:
            return _deep_merge(base, override, variants[0])
        if len(variants) > 1:
            for v in variants:
                merged = _deep_merge(base, override, v)
                if _is_valid_config(v, merged):
                    return merged
            return copy.deepcopy(dict(override))

    dc_fields = (
        {
            (f.default.alias if isinstance(f.default, FieldInfo) else None)
            or f.name: f.type
            for f in fields(tp)
        }
        if isinstance(tp, type) and is_dataclass(tp)
        else {}
    )

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        result[key] = _deep_merge(result.get(key), value, dc_fields.get(key))

    return result


def _dotlist_to_dict(dotlist: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in dotlist:
        if "=" not in entry:
            raise ValueError(
                f"Invalid CLI override '{entry}': expected the form key=value."
            )
        key, _, raw_value = entry.partition("=")
        if not key:
            raise ValueError(f"Invalid CLI override '{entry}': key must not be empty.")
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            nxt = target.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ValueError(
                    f"Cannot set nested key '{key}': '{part}' is not a mapping."
                )
            target = nxt
        target[parts[-1]] = load_yaml(raw_value)
    return result
