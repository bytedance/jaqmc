# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import difflib
import inspect
import logging
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from typing import Any, Protocol, cast, dataclass_transform, overload

import serde
import yaml
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import DiffLexer, YamlLexer
from serde.core import field as serde_field

from jaqmc.utils.module_resolver import resolve_object
from jaqmc.utils.wiring import wire
from jaqmc.utils.yaml_format import annotate_yaml_with_sources, dump_yaml

_MISSING = object()
_PRIMITIVE_TYPES = (int, float, str)
logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "config"}
)

# =============================================================================
# Public API: Decorators and field helpers
# =============================================================================


@dataclass_transform(kw_only_default=True)
def configurable_dataclass(cls=None, *, frozen: bool = False, kw_only: bool = True):
    """Decorator for dataclasses that participate in the config system.

    Applies ``@dataclass`` (if needed) and ``@serde(type_check=coerce)``
    so that the class can be used with
    :class:`~jaqmc.utils.config.ConfigManager`. Coerce mode
    auto-converts compatible types (e.g. ``int`` → ``float``) during
    deserialization.

    Returns:
        The decorated class.

    Example::

        @configurable_dataclass
        class MyConfig:
            learning_rate: float = 0.01
            spins: tuple[int, int] = (1, 0)
    """

    def wrap(cls):
        # Checking __dataclass_fields__ is the robust way to test if a class has been
        # processed by `dataclass` transformations. `is_dataclass` will return True
        # for child classes of dataclasses even if themselves have not been processed.
        if "__dataclass_fields__" not in cls.__dict__:
            cls = dataclass(cls, frozen=frozen, kw_only=kw_only)
        return serde.serde(cls, type_check=serde.coerce, deny_unknown_fields=True)

    if cls is None:
        return wrap
    return wrap(cls)


@dataclass
class ModuleConfig:
    """Metadata for a module_config field."""

    default: str
    kwargs: dict[str, Any]


def module_config(default_factory, **kwargs):
    """Create a serde field for polymorphic module configuration.

    This allows a field to be configured with different implementations
    at runtime via YAML/dict configuration.

    Args:
        default_factory: The default class or callable.
        **kwargs: Default keyword arguments for the factory.

    Returns:
        A serde field descriptor.

    Raises:
        ValueError: If default_factory is not importable by its module path.
    """
    default_module = f"{default_factory.__module__}:{default_factory.__name__}"
    if resolve_object(default_module) is not default_factory:
        raise ValueError(
            f'Expected default_factory to be importable via "{default_module}". '
            f"Got {default_factory}."
        )
    base_package = (
        default_module[: default_module.rfind(".")] if "." in default_module else None
    )
    mc = ModuleConfig(default_module, kwargs)

    def deserializer(data):
        if not isinstance(data, dict):
            raise ValueError(
                f"Something when wrong. Internal representations of dataclasses "
                f"should be dicts. Got {type(data)}."
            )
        mod = data.get("module", default_module)
        cls = resolve_object(mod, package=base_package)
        config_data = {k: v for k, v in data.items() if k != "module"}
        merged = deep_merge(kwargs, config_data)
        if not is_dataclass(cls):
            raise ValueError(f"module_config expected dataclasses, not {cls}.")
        return serde.from_dict(cls, merged)

    def serializer(instance):
        if not is_dataclass(instance):
            raise ValueError(
                f"module_config expected dataclasses, not {type(instance)}."
            )
        result = {
            "module": f"{type(instance).__module__}:{type(instance).__name__}",
            **serde.to_dict(instance),
        }
        return result

    def factory():
        # When kwargs are provided, the default_factory must apply them so that
        # pyserde uses the correct defaults when no user config is given.
        if kwargs and is_dataclass(default_factory):
            return serde.from_dict(default_factory, kwargs)
        elif kwargs:
            return default_factory(**kwargs)
        else:
            return default_factory()

    return serde_field(
        default_factory=factory,
        serializer=serializer,
        deserializer=deserializer,
        metadata={"module_config": mc},
    )


# =============================================================================
# Public API: ConfigManager
# =============================================================================


class ConfigManagerLike(Protocol):
    """Protocol implemented by full and scoped configuration managers."""

    @property
    def name(self) -> str:
        """Dot-separated scope prefix for this manager."""
        return ""

    @overload
    def get[ValueT](self, name: str, default: type[ValueT]) -> ValueT: ...
    @overload
    def get[ValueT](self, name: str, default: ValueT) -> ValueT: ...
    def get[ValueT](self, name: str, default: type[ValueT] | ValueT) -> ValueT:
        """Retrieve a configuration value with type safety.

        The supported types are:

        * Primitive: :py:obj:`int`, :py:obj:`float`, :py:obj:`str`
        * Container: :py:obj:`list` and :py:obj:`dict`
        * :external+python:doc:`Dataclass <library/dataclasses>`
        * :py:obj:`~collections.abc.Callable`

        Args:
            name: The configuration key to retrieve.
            default: A default value or a type/class to use as the schema/default.

        Returns:
            The configuration value, in the same type of `default`.

        Type Parameters:
            ValueT: Type inferred from ``default`` and preserved in the return value.

        Raises:
            NotImplementedError: If the type of `default` is not supported.
        """

    @overload
    def get_module[ModuleT: type | Callable](
        self, name: str, default_module: ModuleT
    ) -> ModuleT: ...
    @overload
    def get_module(self, name: str, default_module: str) -> Any: ...
    def get_module(self, name: str, default_module: str | Callable | type = ""):
        """Instantiate a class or function specified in the configuration.

        Args:
            name: Configuration key pointing to the module settings.
            default_module: Default module or its path if not specified in config.

        Returns:
            The initialized object or result of the function call.

        Type Parameters:
            ModuleT: Module/class/callable type preserved when not using string paths.
        """

    def get_collection(
        self,
        name: str,
        defaults: dict[str, str | dict] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Instantiate a collection of modules from configuration.

        Args:
            name: The configuration section name (e.g. "writers").
            defaults: A dictionary of {key: default_module_path} for standard items.
                Users can disable these by setting them to None in config.
            context: Runtime context dictionary for auto-wiring dependencies.

        Returns:
            A dictionary of {key: instantiated_object}.
        """


class ScopedConfigManager(ConfigManagerLike):
    """A thin wrapper that prepends a prefix to all config lookups.

    Created by ``ConfigManager.scoped(prefix)``. All ``get``, ``get_module``,
    and ``get_collection`` calls are forwarded with ``prefix + "." + name``.

    Args:
        cfg: The underlying ConfigManager.
        prefix: Dot-separated prefix prepended to all keys.
    """

    def __init__(self, cfg: ConfigManagerLike, prefix: str):
        self._cfg = cfg
        self._prefix = prefix

    @property
    def name(self) -> str:
        return self._prefix

    def _key(self, name: str) -> str:
        return f"{self.name}.{name}" if self.name else name

    def get[ValueT](self, name: str, default: type[ValueT] | ValueT) -> ValueT:
        return self._cfg.get(self._key(name), default)

    def get_module(self, name: str, default_module: str | type | Callable = ""):
        return self._cfg.get_module(self._key(name), default_module)

    def get_collection(
        self,
        name: str,
        defaults: dict[str, str | dict] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._cfg.get_collection(self._key(name), defaults, context)


class ConfigManager(ConfigManagerLike):
    """Manages configuration by dynamically resolving schema from usage.

    This class holds the user-provided configuration (from YAML/dicts) and
    command-line overrides (dotlist). Unlike traditional config systems that
    validate against a pre-defined schema, `ConfigManager` defines the schema
    dynamically as values are accessed via `get` or `get_module`.

    When a configuration value is requested, the manager uses the provided
    `default` value to infer the expected type/structure (schema). It then
    reads the corresponding value from the stored user config and dotlist,
    merges it with the default, and returns the result.

    When :meth:`finalize` is called, any source path whose prefix was never
    visited is reported as unused, helping detect typos or obsolete entries.

    Args:
        config: A single configuration dictionary or a sequence of
            dictionaries to be merged. These are typically loaded from
            YAML files.
        dotlist: An optional list of command-line overrides in the
            form of "key=value" or "key.subkey=value".
    """

    def __init__(
        self,
        config: dict[str, Any] | Sequence[dict[str, Any]],
        dotlist: list[str] | None = None,
    ):
        self.resolved_config: dict[str, Any] = {}
        self._visited_paths: set[str] = set()
        # source_map stores metadata about where configuration schemas are defined.
        # Key: configuration path (e.g., "system.molecule").
        # Value: tuple(source_file, line_number, docstring).
        # This is used by `to_yaml` to inject helpful comments.
        self.source_map: dict[str, tuple[str, int, str | None]] = {}

        # Merge user-provided configs into one dict (for unused-key detection).
        configs = [config] if isinstance(config, dict) else list(config)
        self._user_config: dict[str, Any] = {}
        for c in configs:
            self._user_config = deep_merge(self._user_config, c)

        # CLI overrides as nested dict (for unused-key detection + merge).
        self._cli_config = _dotlist_to_dict(dotlist or [])

        # Presets (lowest priority), accumulated via use_preset().
        self._presets: dict[str, Any] = {}

        # Pre-merged config: presets + user + CLI (highest priority).
        self._merged = deep_merge(self._user_config, self._cli_config)

    def scoped(self, prefix: str) -> ScopedConfigManager:
        """Return a scoped view that prepends *prefix* to all lookups.

        Args:
            prefix: Dot-separated prefix (e.g. ``"train"``).

        Returns:
            A :class:`ScopedConfigManager` forwarding to this instance.
        """
        return ScopedConfigManager(self, prefix)

    def __str__(self) -> str:
        return self.to_yaml()

    def to_yaml(self, verbose: bool = False) -> str:
        """Convert resolved configuration to YAML string with source comments.

        Args:
            verbose: If True, also includes docstrings of the configuration objects
                in the comments.

        Returns:
            Formatted YAML string with definition comments.
        """
        yaml_str = dump_yaml(self.resolved_config)
        return annotate_yaml_with_sources(yaml_str, self.source_map, verbose)

    def use_preset(self, preset: dict[str, Any]):
        """Add a preset configuration.

        Presets are merged before user-provided configurations, allowing them
        to be overridden by the user.

        Args:
            preset: A dictionary containing preset configuration values.
        """
        self._presets = deep_merge(self._presets, preset)
        merged = deep_merge(self._presets, self._user_config)
        self._merged = deep_merge(merged, self._cli_config)

    def finalize(
        self,
        raise_on_unused: bool = True,
        verbose: bool = False,
        compare_yaml: str | None = None,
    ):
        """Finalize the configuration and log the results.

        This method outputs the resolved configuration to the log and checks for
        any unused configuration keys.

        Args:
            raise_on_unused: If True, raises `SystemExit` if there are any
                unused configuration keys in the input.
            verbose: If True, includes docstrings and source information in
                the logged YAML output.
            compare_yaml: YAML from previous from to log the difference.

        Raises:
            SystemExit: If `raise_on_unused` is True and unused keys are found.
        """
        yaml_content = self.to_yaml(verbose=verbose)
        logger.info(
            "Resolved configurations:\n%s",
            highlight(yaml_content, YamlLexer(), TerminalFormatter())
            if hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
            else yaml_content,
        )
        unused_yaml = _find_unused_yaml_path(self._visited_paths, self._user_config)
        unused_cli = _find_unused_yaml_path(self._visited_paths, self._cli_config)
        if unused_yaml:
            logger.warning(
                "The following configs are specified via YAML/API but not used: %s",
                sorted(unused_yaml),
            )
        if unused_cli:
            logger.warning(
                "The following configs are specified via CLI but not used: %s",
                sorted(unused_cli),
            )
        if (unused_yaml or unused_cli) and raise_on_unused:
            raise SystemExit(
                "Stopping due to invalid configs specified. Please consider using "
                "`raise_on_unused=False` if you are calling `cfg.finalize` manually, "
                "or pass workflow.config.ignore_extra=True if you are using CLI."
            )
        if compare_yaml is not None:
            diff = "\n".join(
                difflib.unified_diff(
                    compare_yaml.splitlines(),
                    self.to_yaml().splitlines(),
                    fromfile="Restored Config",
                    tofile="Current Config",
                    lineterm="",
                )
            )
            if diff:
                logger.info(
                    "Diff with previous run:\n%s",
                    highlight(diff, DiffLexer(), TerminalFormatter())
                    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
                    else diff,
                )
            else:
                logger.info("Current config is the same as that of previous run.")

    @overload
    def get[ValueT](self, name: str, default: type[ValueT]) -> ValueT: ...
    @overload
    def get[ValueT](self, name: str, default: ValueT) -> ValueT: ...
    def get[ValueT](self, name: str, default: type[ValueT] | ValueT) -> ValueT:
        if isinstance(default, _PRIMITIVE_TYPES):
            return cast(ValueT, self._get_primitive(name, default))
        if isinstance(default, (list, dict)):
            return self._get_container(name, default)
        if is_dataclass(default):
            return cast(ValueT, self._get_dataclass(name, default))
        if callable(default):
            return cast(ValueT, self._get_callable(name, default))

        raise NotImplementedError(
            f"Getting config for type {type(default)} is not supported."
        )

    @overload
    def get_module[ModuleT: type | Callable](
        self, name: str, default_module: ModuleT
    ) -> ModuleT: ...
    @overload
    def get_module(self, name: str, default_module: str) -> Any: ...
    def get_module(self, name: str, default_module: str | Callable | type = ""):
        if isinstance(default_module, str):
            module_base = (
                None
                if "." not in default_module
                else default_module[: default_module.rfind(".")]
            )
            module_name = self._get_primitive(f"{name}.module", default_module)
            make_module = resolve_object(module_name, package=module_base)
        else:
            default_module_name = (
                f"{default_module.__module__}:{default_module.__name__}"
            )
            module_base = (
                None
                if "." not in default_module.__module__
                else default_module.__module__[: default_module.__module__.rfind(".")]
            )
            module_name = self._get_primitive(f"{name}.module", default_module_name)
            if module_name != default_module_name:
                make_module = resolve_object(module_name, package=module_base)
            else:
                make_module = default_module
        if is_dataclass(make_module):
            return self._get_dataclass(name, make_module, module=module_name)
        return self._get_callable(name, make_module, module=module_name)

    def get_collection(
        self,
        name: str,
        defaults: dict[str, str | dict] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        user_config = self._extract_relevant(name)
        default_dict = {
            k: {"module": v} if isinstance(v, str) else v
            for k, v in (defaults or {}).items()
        }
        module_configs = deep_merge(
            default_dict, user_config if user_config is not _MISSING else {}
        )
        result = {}
        for k, v in module_configs.items():
            if v is None:
                continue
            module_path = v["module"]
            make_module = resolve_object(module_path)
            if is_dataclass(make_module):
                default_item = default_dict.get(k, {})
                default_config = {
                    key: value
                    for key, value in default_item.items()
                    if key != "module" and default_item.get("module") == module_path
                }
                default_obj = (
                    serde.from_dict(make_module, default_config)
                    if default_config
                    else make_module
                )
                instance = self._get_dataclass(
                    f"{name}.{k}", default_obj, module=module_path
                )
                wire(instance, **context)
                result[k] = instance
            else:
                result[k] = self._get_callable(
                    f"{name}.{k}",
                    make_module,
                    module=module_path,
                )(**context)
        return result

    def _register_source(self, name: str, obj: Any):
        """Record the source location and docstring of a configuration schema."""
        if name in self.source_map or name.endswith(".module"):
            return

        def is_user_defined(o):
            t = o if isinstance(o, type) or callable(o) else type(o)
            return t.__module__ != "builtins"

        found_definition = False
        if is_user_defined(obj):
            try:
                target = obj
                if not isinstance(target, type) and not callable(target):
                    target = type(target)

                source_file = inspect.getsourcefile(target)
                if source_file:
                    source_lines = inspect.getsourcelines(target)
                    lineno = source_lines[1]
                    doc = inspect.getdoc(target)
                    self.source_map[name] = (source_file, lineno, doc)
                    found_definition = True
            except (OSError, TypeError):
                pass

        if not found_definition:
            current_file = __file__
            stack = inspect.stack()
            for frame in stack:
                if frame.filename != current_file:
                    self.source_map[name] = (frame.filename, frame.lineno, None)
                    break

    def _extract_relevant(self, name: str) -> Any:
        self._visited_paths.add(name)
        return _get_path(self._merged, name.split("."), default=_MISSING)

    def _get_primitive[PrimitiveT: str | int | float](
        self, name: str, default: PrimitiveT
    ) -> PrimitiveT:
        user_val = self._extract_relevant(name)
        if user_val is not _MISSING:
            result = type(default)(user_val) if default is not None else user_val
        else:
            result = default
        _set_path(self.resolved_config, name, result)
        return cast(PrimitiveT, result)

    def _get_container[ContainerT: list | dict](
        self, name: str, default: ContainerT
    ) -> ContainerT:
        self._register_source(name, default)
        user_val = self._extract_relevant(name)
        result: Any
        if user_val is not _MISSING:
            if isinstance(default, dict) and isinstance(user_val, dict):
                result = deep_merge(default, user_val)
            else:
                result = user_val
        else:
            result = default
        _set_path(self.resolved_config, name, result)
        return cast(ContainerT, result)

    def _get_dataclass[DataclassT](
        self, name: str, default: DataclassT, module: str = ""
    ) -> DataclassT:
        self._register_source(name, default)
        user_config = self._extract_relevant(name)
        cls = default if inspect.isclass(default) else type(default)

        # When default is an instance, use its field values as the base.
        # This ensures instance-level overrides (e.g., batch_size=None)
        # are respected even when the class default differs.
        base_config = serde.to_dict(default) if not inspect.isclass(default) else {}

        config_data = user_config if user_config is not _MISSING else {}
        if isinstance(config_data, dict):
            config_data = {k: v for k, v in config_data.items() if k != "module"}
        else:
            config_data = {}

        merged = deep_merge(base_config, config_data)
        try:
            result = serde.from_dict(cls, merged)
        except serde.SerdeError as e:
            raise serde.SerdeError(f"Invalid config at '{name}': {e}") from None

        if not inspect.isclass(default):
            _copy_runtime_fields(default, result)

        resolved = serde.to_dict(result)
        if module:
            resolved = {"module": module, **resolved}
        _set_path(self.resolved_config, name, resolved)

        return cast(DataclassT, result)

    def _get_callable[CallableT: Callable](
        self,
        name: str,
        default: CallableT,
        module: str = "",
        override_default: dict[str, Any] | None = None,
    ) -> CallableT:
        self._register_source(name, default)
        user_config = self._extract_relevant(name)
        defaults = _callable_defaults(default, module, override_default)
        user = user_config if user_config is not _MISSING else {}
        merged = deep_merge(defaults, user)
        _set_path(self.resolved_config, name, merged)
        kw = {k: v for k, v in merged.items() if k != "module"}
        return _safe_kw_partial(default, **kw)


# =============================================================================
# Public helpers: Dict utilities
# =============================================================================


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. override wins for leaf values.

    Returns:
        A new merged dict.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# =============================================================================
# Private helpers
# =============================================================================


def _set_path(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _get_path(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    if len(path) == 1:
        return data.get(path[0], default)
    key = path[0]
    if key not in data or not isinstance(data[key], dict):
        return default
    return _get_path(data[key], path[1:], default=default)


def _dotlist_to_dict(dotlist: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in dotlist:
        key, _, raw_value = entry.partition("=")
        _set_path(result, key, yaml.safe_load(raw_value))
    return result


def _safe_kw_partial(f, **config):
    """Returns a partial that filters out kwargs not accepted by f."""
    params = inspect.signature(f).parameters.values()
    has_var_keyword = any(param.kind == param.VAR_KEYWORD for param in params)
    kw_list = [param.name for param in params]

    @wraps(f)
    def wrapped(*args, **kwargs):
        if has_var_keyword:
            return f(*args, **(config | kwargs))
        merged = config | {k: v for k, v in kwargs.items() if k in kw_list}
        return f(*args, **merged)

    return wrapped


def _callable_defaults(
    func: Callable,
    module: str = "",
    override_default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Returns default parameter values from a callable's signature."""
    override_default = override_default or {}
    defaults: dict[str, Any] = {}
    if module:
        defaults["module"] = module
    for param in inspect.signature(func).parameters.values():
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        if param.default is inspect.Parameter.empty:
            continue
        defaults[param.name] = override_default.get(param.name, param.default)
    return defaults


def _copy_runtime_fields(source: Any, target: Any) -> None:
    """Copy serde_skip field values from source to target dataclass."""
    for fld in fields(target):
        if fld.metadata.get("serde_skip"):
            src_val = getattr(source, fld.name, None)
            if src_val is not None:
                setattr(target, fld.name, src_val)
        elif is_dataclass(val := getattr(target, fld.name)):
            src_val = getattr(source, fld.name, None)
            if src_val is not None and is_dataclass(src_val):
                _copy_runtime_fields(src_val, val)


def _find_unused_yaml_path(
    visited_paths: Iterable[str], config: dict[str, Any]
) -> set[str]:
    """Returns config paths that were specified but never accessed."""
    unused_paths = set()
    for k, v in config.items():
        if k in visited_paths:
            continue
        if not isinstance(v, dict):
            unused_paths.add(k)
            continue
        visited_subpaths = [
            path[len(k) + 1 :] for path in visited_paths if path.startswith(f"{k}.")
        ]
        unused_paths |= {
            f"{k}.{s}" for s in _find_unused_yaml_path(visited_subpaths, v)
        }
    return unused_paths
