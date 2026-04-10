# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Sphinx directives for rendering contextual config defaults in docs.

Usage in MyST markdown::

    ```{eval-rst}
    .. config-context::
       :preset: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow.default_preset
    ```

    ```{eval-rst}
    .. config-defaults:: jaqmc.optimizer.kfac.kfac.KFACOptimizer
       :prefix: train.optim
    ```

Each field renders as a compact stacked row showing the key, effective
default, type, and a short description.
"""

import enum
import importlib
import inspect
import re
from collections.abc import Callable
from dataclasses import MISSING, dataclass, is_dataclass
from dataclasses import fields as dc_fields
from typing import Any, ClassVar, cast

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util.docutils import SphinxDirective

from jaqmc.utils.module_resolver import resolve_object

_NO_OVERRIDE = object()
_OptionSpec = dict[str, Callable[[str], Any]]


class ConfigDefaultsSeparatorNode(nodes.General, nodes.Element):
    """Separator between rendered config-default entries."""


class ConfigDefaultsNestedNode(nodes.General, nodes.Element):
    """Nested block for swappable-component options."""


@dataclass
class ConfigDefaultEntry:
    """One rendered entry in a compact config-defaults list."""

    key: str
    default_str: str
    type_str: str | None = None
    description: str | None = None
    children: list["ConfigDefaultEntry"] | None = None
    classes: tuple[str, ...] = ()
    anchor_id: str | None = None
    ref_label: str | None = None
    anchor_scope: str | None = None


def _save_field(result: dict[str, str], name: str | None, desc: list[str]) -> None:
    """Save a parsed field into the result dict."""
    if name is not None:
        text = "\n".join(desc).strip()
        if text:
            result[name] = text


def _parse_args_section(docstring: str | None) -> dict[str, str]:
    """Extract field descriptions from a Google-style Args: section.

    Returns:
        Mapping of field name to description text.
    """
    if not docstring:
        return {}

    result: dict[str, str] = {}

    current_name: str | None = None
    current_desc: list[str] = []
    for line in str(GoogleDocstring(docstring)).splitlines():
        if line.startswith(":param "):
            match = re.match(r"^:param\s+(\w+)\s*:\s*(.*)$", line)
            if not match:
                continue
            _save_field(result, current_name, current_desc)
            current_name = match.group(1)
            current_desc = [match.group(2)] if match.group(2) else []
            continue

        if current_name is None:
            continue

        if line.startswith(":"):
            _save_field(result, current_name, current_desc)
            current_name = None
            current_desc = []
            continue

        if not line.strip():
            current_desc.append("")
            continue

        current_desc.append(line)

    _save_field(result, current_name, current_desc)
    return result


def _collect_dataclass_descriptions(obj: Any) -> dict[str, str]:
    """Collect docstring descriptions for dataclass fields across the MRO.

    Returns:
        Mapping from field name to description text.
    """
    descriptions: dict[str, str] = {}
    for base in reversed(obj.__mro__):
        if is_dataclass(base):
            descriptions.update(_parse_args_section(inspect.getdoc(base)))
    return descriptions


def _compact_description(text: str | None) -> str | None:
    """Collapse a field description into a short inline sentence.

    Returns:
        First sentence of the first paragraph, or ``None`` when unavailable.
    """
    if not text:
        return None

    first_paragraph = text.strip().split("\n\n", 1)[0]
    compact = " ".join(line.strip() for line in first_paragraph.splitlines()).strip()
    if not compact:
        return None

    sentence_break = re.search(r"(?<=[.!?])\s+(?=[A-Z(])", compact)
    if sentence_break:
        compact = compact[: sentence_break.start()].strip()
    return compact or None


def _compact_resolved_description(text: str | None, mod: object) -> str | None:
    """Return a compact description with local references resolved."""
    compact = _compact_description(text)
    return _resolve_refs(compact, mod) if compact else None


def _type_name(t) -> str:
    """Return a plain-text type name without RST markup.

    Recursively formats generic types, unions, etc. into a readable string
    like ``list[int | None]``.
    """
    if t is inspect.Parameter.empty:
        return ""

    if isinstance(t, str):
        return t

    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", ())

    # X | Y syntax or typing.Union
    import typing

    if origin is type(int | str) or origin is typing.Union:
        return " | ".join(_type_name(a) for a in args)

    # list[X], tuple[X, Y], etc.
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            inner = ", ".join(_type_name(a) for a in args)
            return f"{origin_name}[{inner}]"
        return origin_name

    # NoneType
    if t is type(None):
        return "None"

    return getattr(t, "__name__", None) or getattr(t, "_name", None) or str(t)


def _format_type_str(param: inspect.Parameter) -> str:
    """Format parameter type annotation as inline code, or empty string.

    Returns:
        RST inline-code string, or empty string if no annotation.
    """
    if param.annotation is inspect.Parameter.empty:
        return ""
    type_name = _type_name(param.annotation)
    return f"``{type_name}``" if type_name else ""


def _format_compact_type(field_type, mc) -> str | None:
    """Format a field type for compact contextual defaults.

    Returns:
        Compact RST-formatted type string, or ``None`` if unavailable.
    """
    if mc is not None:
        return "swappable"
    name = _type_name(field_type)
    return f"``{name}``" if name else None


def _format_default_str(param: inspect.Parameter) -> str:
    """Format parameter default value as RST markup.

    Returns:
        RST string showing the default, or "*(required)*" if none.
    """
    return _format_value(
        param.default,
        required=param.default is inspect.Parameter.empty,
    )


def _format_value(value: Any, *, required: bool = False) -> str:
    """Format an arbitrary default value as RST markup.

    Returns:
        RST-formatted default value string.
    """
    if required:
        return "*(required)*"
    if isinstance(value, enum.Enum):
        fq = _enum_target(value)
        return f":py:attr:`{value.name} <{fq}>`" if fq else f"``{value.name}``"
    return f"``{value!r}``"


def _enum_target(value: enum.Enum) -> str | None:
    """Return the best public cross-reference target for an enum member."""
    cls = type(value)
    candidates = [cls.__module__]
    module_name = cls.__module__

    while "." in module_name:
        module_name = module_name.rsplit(".", 1)[0]
        candidates.append(module_name)

    resolved_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            mod = importlib.import_module(candidate)
        except ImportError:
            continue
        if getattr(mod, cls.__name__, None) is cls:
            resolved_candidates.append(candidate)

    if resolved_candidates:
        best = min(resolved_candidates, key=lambda name: (name.count("."), len(name)))
        return f"{best}.{cls.__qualname__}.{value.name}"

    return f"{cls.__module__}.{cls.__qualname__}.{value.name}"


def _resolve_refs(text: str, mod: object) -> str:
    """Resolve short ``:attr:`` and ``:class:`` references to full paths.

    Looks up unqualified class names (e.g. ``LayerNormMode``) in the
    module's namespace and replaces them with fully qualified paths
    prefixed with ``~`` so Sphinx displays the short name but links
    to the correct target.

    Returns:
        Text with short references replaced by fully qualified paths.
    """

    def _replace(m: re.Match) -> str:
        role = m.group(1)
        ref = m.group(2)
        # Already qualified (contains a dot-separated module path).
        cls_name = ref.split(".")[0] if "." in ref else ref
        obj = getattr(mod, cls_name, None)
        if obj is not None:
            fq_module = getattr(obj, "__module__", None)
            if fq_module:
                return f":{role}:`~{fq_module}.{ref}`"
        return m.group(0)

    return re.sub(r":(\w+):`([^`]+)`", _replace, text)


_SWAPPABLE_PREFIXES = {
    "system",
    "wf",
    "train.optim",
    "pretrain.optim",
    "train.sampler",
    "pretrain.sampler",
}


def _slugify(text: str) -> str:
    """Return a lowercase slug for HTML ids."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _anchor_slug(key: str, scope: str | None = None) -> str:
    """Return a stable HTML anchor for a config key."""
    if scope:
        if "." in key:
            prefix, leaf = key.rsplit(".", 1)
            slug = f"{_slugify(prefix)}-{_slugify(scope)}-{_slugify(leaf)}"
        else:
            slug = f"{_slugify(key)}-{_slugify(scope)}"
    else:
        slug = _slugify(key)
    return f"cfg-{slug}" if slug else "cfg"


def _label_slug(docname: str, anchor_id: str) -> str:
    """Return a stable cross-reference label for a config key."""
    return f"{docname.replace('/', '-')}-{anchor_id}"


def _strip_common_suffixes(name: str) -> str:
    """Strip conventional config object suffixes from a class or function name.

    Returns:
        The name without a known config-related suffix.
    """
    for suffix in (
        "Wavefunction",
        "Optimizer",
        "Sampler",
        "Writer",
        "Config",
        "WorkStageConfig",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _scope_from_object(obj: Any) -> str | None:
    """Derive an implementation scope from a config object.

    Returns:
        A slugified implementation name, or ``None`` if unavailable.
    """
    name = getattr(obj, "__name__", None) or getattr(obj, "__qualname__", "")
    stripped = _strip_common_suffixes(str(name))
    if stripped:
        return _slugify(stripped)
    return None


def _scope_from_module_name(module_name: str) -> str | None:
    """Derive an implementation scope from a module_config target string.

    Returns:
        A slugified implementation name parsed from the module path.
    """
    if ":" in module_name:
        return _slugify(module_name.rsplit(":", 1)[1])
    return _slugify(module_name.rsplit(".", 1)[1])


def _default_anchor_scope(obj: Any, prefix: str) -> str | None:
    """Return the default scope prefix for swappable config sections."""
    if prefix not in _SWAPPABLE_PREFIXES:
        return None
    return _scope_from_object(obj)


def _merge_anchor_scopes(parent: str | None, child: str | None) -> str | None:
    """Combine parent and child scopes for nested swappable config entries.

    Returns:
        The merged scope string, or whichever scope is available.
    """
    if parent and child:
        return f"{parent}-{child}"
    return child or parent


def _resolve_qualified(name: str) -> Any:
    """Resolve a dotted ``module.attr`` path, including attribute chains.

    Returns:
        The resolved object.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    parts = name.split(".")
    last_exc: Exception | None = None
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            obj: Any = importlib.import_module(module_name)
        except ImportError as exc:
            last_exc = exc
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_exc = exc
            continue
        return obj
    raise ValueError(f"Cannot resolve {name}: {last_exc}")


def _resolve_preset(name: str) -> dict[str, Any]:
    """Resolve and call a preset provider.

    Returns:
        The preset dictionary returned by the provider.

    Raises:
        TypeError: If the provider is not callable or does not return a dict.
    """
    preset = _resolve_qualified(name)
    if not callable(preset):
        raise TypeError(f"Preset provider {name} is not callable.")
    value = preset()
    if not isinstance(value, dict):
        raise TypeError(
            f"Preset provider {name} must return a dict, got {type(value)}."
        )
    return value


def _get_prefixed_override(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return the nested override dict at ``prefix``."""
    if not prefix:
        return data
    current: Any = data
    for part in prefix.split("."):
        if not isinstance(current, dict):
            return {}
        current = current.get(part, {})
    return current if isinstance(current, dict) else {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dicts, with ``override`` winning leaf values.

    Returns:
        A new merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _module_config_package(default_module: str) -> str | None:
    """Return the package used for relative resolution of ``module_config``."""
    if "." not in default_module:
        return None
    return default_module[: default_module.rfind(".")]


def _format_module_value(module_name: str, package: str | None = None) -> str:
    """Format a module path for compact display in contextual defaults.

    Returns:
        RST-formatted module value string.
    """
    try:
        obj = resolve_object(module_name, package=package)
    except Exception:
        display = (
            module_name
            if ":" not in module_name or "." not in module_name.split(":", 1)[0]
            else module_name.rsplit(":", 1)[1]
        )
        return f"``{display}``"

    target = f"{obj.__module__}.{obj.__qualname__}"
    display = getattr(obj, "__name__", obj.__qualname__.split(".")[-1])
    return f":py:obj:`{display} <{target}>`"


def _format_default(field, mc, override: Any = _NO_OVERRIDE) -> str:
    """Format a field's default value for display.

    Returns:
        RST-formatted default value string.
    """
    if override is not _NO_OVERRIDE:
        if mc is not None and isinstance(override, dict):
            module_name = override.get("module", mc.default)
            target = module_name.replace(":", ".")
            class_short = module_name.rsplit(":", 1)[1]
            kwargs = _merge_dicts(
                mc.kwargs, {k: v for k, v in override.items() if k != "module"}
            )
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            display = (
                f"{class_short}({kwargs_str})" if kwargs_str else f"{class_short}()"
            )
            return f":py:obj:`{display} <{target}>`"
        return _format_value(override)

    if mc is not None:
        target = mc.default.replace(":", ".")
        class_short = mc.default.rsplit(":", 1)[1]
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in mc.kwargs.items())
        display = f"{class_short}({kwargs_str})" if kwargs_str else f"{class_short}()"
        return f":py:obj:`{display} <{target}>`"

    if field.default is not MISSING:
        return _format_value(field.default)
    if field.default_factory is not MISSING:
        try:
            val = field.default_factory()
            return _format_value(val)
        except Exception:
            return "*(factory)*"
    return "*(required)*"


class ConfigContext(SphinxDirective):
    """Store page-scoped config rendering context for later directives."""

    has_content = False
    required_arguments = 0
    option_spec: ClassVar[_OptionSpec] = {
        "preset": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        self.env.temp_data["config-context"] = {
            key: value for key, value in self.options.items()
        }
        return []


class ConfigDirectiveBase(SphinxDirective):
    """Shared plumbing for config-rendering directives."""

    def _report_error(self, message: str) -> list[nodes.Node]:
        error = self.state_machine.reporter.error(
            message,
            nodes.literal_block(self.block_text, self.block_text),
            line=self.lineno,
        )
        return [error]

    def _get_context_preset(self) -> str | None:
        if "preset" in self.options:
            return self.options["preset"]
        context = self.env.temp_data.get("config-context", {})
        if not isinstance(context, dict):
            return None
        preset = context.get("preset")
        return preset if isinstance(preset, str) else None

    def _resolve_target(self, qualified: str) -> tuple[Any, object] | list[nodes.Node]:
        try:
            obj = _resolve_qualified(qualified)
            mod = importlib.import_module(getattr(obj, "__module__", "__main__"))
        except (ImportError, AttributeError, ValueError) as e:
            return self._report_error(f"Cannot import {qualified}: {e}")
        return obj, mod

    def _resolve_overrides(
        self, qualified: str, prefix: str
    ) -> dict[str, Any] | list[nodes.Node]:
        preset_name = self._get_context_preset()
        try:
            return _get_prefixed_override(
                _resolve_preset(preset_name) if preset_name else {},
                prefix,
            )
        except (TypeError, ValueError, ImportError, AttributeError) as e:
            return self._report_error(f"Cannot resolve preset for {qualified}: {e}")

    def _collect_exclude_names(self, obj: Any) -> set[str]:
        exclude_names: set[str] = set()
        if "exclude-inherited" in self.options:
            for base in obj.__mro__[1:]:
                if is_dataclass(base):
                    exclude_names.update(f.name for f in dc_fields(base))
        if "exclude-fields" in self.options:
            exclude_names.update(
                field.strip()
                for field in self.options["exclude-fields"].split(",")
                if field.strip()
            )
        return exclude_names


class ConfigDefaults(ConfigDirectiveBase):
    """Render effective contextual defaults as compact stacked rows."""

    has_content = False
    required_arguments = 1
    option_spec: ClassVar[_OptionSpec] = {
        "prefix": directives.unchanged,
        "preset": directives.unchanged,
        "exclude-fields": directives.unchanged,
        "exclude-inherited": directives.flag,
    }

    def _module_group_description(
        self, description: str | None, field_key: str, module_str: str
    ) -> str:
        """Build the explanatory line for a swappable component block.

        Returns:
            Compact inline description for the group header.
        """
        parts: list[str] = []
        if description:
            parts.append(description.rstrip("."))
        parts.append(
            "Swappable component; the nested keys below are the options for the "
            f"current module {module_str} and change when "
            f"``{field_key}.module`` changes."
        )
        return ". ".join(parts)

    def _meta_text(self, entry: ConfigDefaultEntry) -> str:
        """Build the compact metadata line shown under the key.

        Returns:
            Inline metadata string containing default and type.
        """
        parts = [f"Default: {entry.default_str}"]
        if entry.type_str:
            parts.append(f"Type: {entry.type_str}")
        return " · ".join(parts)

    def _description(
        self, descriptions: dict[str, str], name: str, mod: object
    ) -> str | None:
        """Return the compact rendered description for a field or parameter."""
        return _compact_resolved_description(descriptions.get(name), mod)

    def _make_item(
        self, entry: ConfigDefaultEntry
    ) -> tuple[nodes.container, list[nodes.Node]]:
        item = nodes.container(classes=["config-defaults-item", *entry.classes])
        messages: list[nodes.Node] = []

        if entry.anchor_id and entry.ref_label:
            target = nodes.target(
                "",
                "",
                ids=[entry.anchor_id],
            )
            self.state.document.note_explicit_target(target)
            self.env.domains.standard_domain.note_object(
                "cfgkey",
                entry.ref_label,
                entry.anchor_id,
                location=target,
            )
            item += target

        key_line = nodes.paragraph(classes=["config-defaults-header"])
        key_node = nodes.strong(entry.key, entry.key, classes=["config-defaults-key"])
        key_line += key_node
        item += key_line

        meta = nodes.paragraph(classes=["config-defaults-meta"])
        meta_nodes, meta_messages = self.state.inline_text(
            self._meta_text(entry), self.lineno
        )
        meta.extend(meta_nodes)
        item += meta
        messages.extend(meta_messages)

        if entry.description:
            body = nodes.paragraph(classes=["config-defaults-meaning"])
            desc_nodes, desc_messages = self.state.inline_text(
                entry.description, self.lineno
            )
            body.extend(desc_nodes)
            item += body
            messages.extend(desc_messages)

        if entry.children:
            nested_wrapper = ConfigDefaultsNestedNode(
                classes=["config-defaults-nested"]
            )
            for child in entry.children:
                child_item, child_messages = self._make_item(
                    ConfigDefaultEntry(
                        key=child.key,
                        default_str=child.default_str,
                        type_str=child.type_str,
                        description=child.description,
                        children=child.children,
                        classes=(*child.classes, "config-defaults-item-nested"),
                        anchor_id=child.anchor_id,
                        ref_label=child.ref_label,
                        anchor_scope=child.anchor_scope,
                    )
                )
                nested_wrapper += child_item
                messages.extend(child_messages)
            item += nested_wrapper

        return item, messages

    def _assign_anchor_ids(
        self, entries: list[ConfigDefaultEntry], used: dict[str, int] | None = None
    ) -> None:
        """Assign stable, unique HTML anchor ids to rendered entries."""
        if used is None:
            used = {}
        docname = self.env.current_document.docname
        for entry in entries:
            base = _anchor_slug(entry.key, entry.anchor_scope)
            count = used.get(base, 0)
            used[base] = count + 1
            entry.anchor_id = base if count == 0 else f"{base}-{count + 1}"
            entry.ref_label = _label_slug(docname, entry.anchor_id)
            if entry.children:
                self._assign_anchor_ids(entry.children, used)

    def _make_list(self, lines: list[ConfigDefaultEntry]) -> list[nodes.Node]:
        self._assign_anchor_ids(lines)
        wrapper = nodes.container(classes=["config-defaults-list"])
        messages: list[nodes.Node] = []
        for index, entry in enumerate(lines):
            if index:
                wrapper += ConfigDefaultsSeparatorNode()
            item, item_messages = self._make_item(entry)
            wrapper += item
            messages.extend(item_messages)
        return [wrapper, *messages]

    def _collect_module_config_item(
        self,
        field,
        field_key: str,
        module_config,
        override: Any,
        descriptions: dict[str, str],
        mod: object,
        anchor_scope: str | None,
    ) -> ConfigDefaultEntry:
        """Collect a grouped entry for a swappable ``module_config`` field.

        Returns:
            Parent entry with nested current-module options.
        """
        module_name = (
            override.get("module", module_config.default)
            if isinstance(override, dict)
            else module_config.default
        )
        package = _module_config_package(module_config.default)
        module_default_str = _format_module_value(module_name, package=package)
        nested_obj = resolve_object(module_name, package=package)
        nested_overrides = (
            {k: v for k, v in override.items() if k != "module"}
            if isinstance(override, dict)
            else {}
        )
        nested_items: list[ConfigDefaultEntry] = [
            ConfigDefaultEntry(
                key=f"{field_key}.module",
                default_str=module_default_str,
                type_str="module path",
                description="Select the implementation used for this component.",
                anchor_scope=anchor_scope,
            )
        ]
        nested_scope = _merge_anchor_scopes(
            anchor_scope,
            _scope_from_module_name(module_name),
        )
        if is_dataclass(nested_obj):
            nested_items.extend(
                self._collect_dataclass_items(
                    nested_obj,
                    field_key,
                    nested_overrides,
                    importlib.import_module(nested_obj.__module__),
                    anchor_scope=nested_scope,
                )
            )
        return ConfigDefaultEntry(
            key=field_key,
            default_str=module_default_str,
            type_str="swappable",
            description=self._module_group_description(
                self._description(descriptions, field.name, mod),
                field_key,
                module_default_str,
            ),
            children=nested_items,
            classes=("config-defaults-item-group",),
            anchor_scope=anchor_scope,
        )

    def _collect_callable_items(
        self,
        fn: object,
        prefix: str,
        mod: object,
        overrides: dict[str, Any],
        anchor_scope: str | None = None,
    ) -> list[ConfigDefaultEntry]:
        items: list[ConfigDefaultEntry] = []
        callable_fn = cast(Callable[..., Any], fn)
        descriptions = _parse_args_section(inspect.getdoc(fn))
        for param in inspect.signature(callable_fn).parameters.values():
            override = overrides.get(param.name, _NO_OVERRIDE)
            default_str = (
                _format_value(override)
                if override is not _NO_OVERRIDE
                else _format_default_str(param)
            )
            key = f"{prefix}.{param.name}" if prefix else param.name
            items.append(
                ConfigDefaultEntry(
                    key=key,
                    default_str=default_str,
                    type_str=_format_type_str(param) or None,
                    description=self._description(descriptions, param.name, mod),
                    anchor_scope=anchor_scope,
                )
            )
        return items

    def _collect_dataclass_items(
        self,
        cls: Any,
        prefix: str,
        overrides: dict[str, Any],
        mod: object,
        exclude_names: set[str] | None = None,
        anchor_scope: str | None = None,
    ) -> list[ConfigDefaultEntry]:
        items: list[ConfigDefaultEntry] = []
        descriptions = _collect_dataclass_descriptions(cls)
        for field in dc_fields(cls):
            if exclude_names and field.name in exclude_names:
                continue
            if field.metadata.get("runtime") or field.metadata.get("serde_skip"):
                continue

            override = overrides.get(field.name, _NO_OVERRIDE)
            field_key = f"{prefix}.{field.name}" if prefix else field.name
            module_config = field.metadata.get("module_config")
            field_type = field.type if not isinstance(field.type, str) else None

            if module_config is not None:
                items.append(
                    self._collect_module_config_item(
                        field,
                        field_key,
                        module_config,
                        override,
                        descriptions,
                        mod,
                        anchor_scope,
                    )
                )
                continue

            if (
                field_type is not None
                and isinstance(field_type, type)
                and is_dataclass(field_type)
            ):
                items.extend(
                    self._collect_dataclass_items(
                        field_type,
                        field_key,
                        override if isinstance(override, dict) else {},
                        importlib.import_module(field_type.__module__),
                        anchor_scope=anchor_scope,
                    )
                )
                continue

            default_str = _format_default(field, module_config, override=override)
            items.append(
                ConfigDefaultEntry(
                    key=field_key,
                    default_str=default_str,
                    type_str=_format_compact_type(field.type, module_config),
                    description=self._description(descriptions, field.name, mod),
                    anchor_scope=anchor_scope,
                )
            )
        return items

    def run(self) -> list[nodes.Node]:
        qualified = self.arguments[0]
        target = self._resolve_target(qualified)
        if isinstance(target, list):
            return target
        obj, mod = target

        prefix = self.options.get("prefix", "")
        anchor_scope = _default_anchor_scope(obj, prefix)
        overrides = self._resolve_overrides(qualified, prefix)
        if isinstance(overrides, list):
            return overrides

        if is_dataclass(obj):
            lines = self._collect_dataclass_items(
                obj,
                prefix,
                overrides,
                mod,
                exclude_names=self._collect_exclude_names(obj),
                anchor_scope=anchor_scope,
            )
        elif callable(obj):
            lines = self._collect_callable_items(
                obj, prefix, mod, overrides, anchor_scope=anchor_scope
            )
        else:
            return self._report_error(
                f"{qualified} is neither a dataclass nor a callable"
            )

        if not lines:
            return []

        return self._make_list(lines)


def setup(app: Sphinx):
    def visit_config_defaults_nested_html(self, node: ConfigDefaultsNestedNode):
        self.body.append('<div class="config-defaults-nested">')

    def depart_config_defaults_nested_html(self, node: ConfigDefaultsNestedNode):
        self.body.append("</div>")

    def visit_config_defaults_nested_latex(self, node: ConfigDefaultsNestedNode):
        self.body.append("\n\\begin{quote}\n")

    def depart_config_defaults_nested_latex(self, node: ConfigDefaultsNestedNode):
        self.body.append("\n\\end{quote}\n")

    def visit_config_defaults_nested_noop(self, node: ConfigDefaultsNestedNode):
        pass

    def depart_config_defaults_nested_noop(self, node: ConfigDefaultsNestedNode):
        pass

    def visit_config_defaults_separator_html(self, node: ConfigDefaultsSeparatorNode):
        self.body.append('<hr class="config-defaults-separator" />')
        raise nodes.SkipNode

    def visit_config_defaults_separator_latex(self, node: ConfigDefaultsSeparatorNode):
        self.body.append(
            "\n\\par\\smallskip"
            "{\\color[gray]{0.72}\\hrule height 0.25pt}"
            "\\smallskip\\par\n"
        )
        raise nodes.SkipNode

    def visit_config_defaults_separator_noop(self, node: ConfigDefaultsSeparatorNode):
        raise nodes.SkipNode

    app.add_node(
        ConfigDefaultsNestedNode,
        html=(visit_config_defaults_nested_html, depart_config_defaults_nested_html),
        latex=(visit_config_defaults_nested_latex, depart_config_defaults_nested_latex),
        text=(visit_config_defaults_nested_noop, depart_config_defaults_nested_noop),
        man=(visit_config_defaults_nested_noop, depart_config_defaults_nested_noop),
        texinfo=(
            visit_config_defaults_nested_noop,
            depart_config_defaults_nested_noop,
        ),
    )
    app.add_node(
        ConfigDefaultsSeparatorNode,
        html=(visit_config_defaults_separator_html, None),
        latex=(visit_config_defaults_separator_latex, None),
        text=(visit_config_defaults_separator_noop, None),
        man=(visit_config_defaults_separator_noop, None),
        texinfo=(visit_config_defaults_separator_noop, None),
    )
    app.add_object_type("cfgkey", "cfgkey", objname="config key")
    app.add_directive("config-context", ConfigContext)
    app.add_directive("config-defaults", ConfigDefaults)
    return {"version": "0.1", "parallel_read_safe": True}
