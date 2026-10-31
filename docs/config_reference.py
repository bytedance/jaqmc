# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Sphinx directives for rendering contextual config defaults in docs.

Usage in MyST markdown::

    ```{eval-rst}
    .. config-defaults:: jaqmc.optimizer.kfac.kfac.KFACOptimizer
       :prefix: train.optim
       :scope: KFAC
    ```

Each field renders as a compact stacked row showing the key,
default, type, and a short description.
"""

import enum
import importlib
import inspect
import re
import textwrap
from collections.abc import Callable, Iterator
from collections.abc import Set as AbstractSet
from dataclasses import MISSING, dataclass, is_dataclass, replace
from dataclasses import fields as dc_fields
from typing import (
    Annotated,
    Any,
    ClassVar,
    TypeAliasType,
    cast,
    get_args,
    get_origin,
)

from docutils import nodes
from docutils.nodes import Element
from docutils.parsers.rst import directives
from pydantic.fields import FieldInfo
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx_autodoc_typehints import format_annotation

from jaqmc.estimator.base import Estimator, EstimatorCollection
from jaqmc.utils.config import (
    ModuleConfig,
    config_to_dict,
    is_configurable_dataclass,
    materialize_config_default,
)
from jaqmc.utils.yaml_format import dump_yaml

logger = logging.getLogger(__name__)

_OptionSpec = dict[str, Callable[[str], Any]]


def _has_module_config(tp: Any) -> bool:
    return get_origin(tp) is Annotated and any(
        isinstance(meta, ModuleConfig) for meta in get_args(tp)[1:]
    )


class ConfigDefaultsSeparatorNode(nodes.General, nodes.Element):
    """Separator between rendered config-default entries."""


class ConfigDefaultsNestedNode(nodes.General, nodes.Element):
    """Nested block for swappable-component options."""


@dataclass
class ConfigDefaultEntry:
    """One rendered entry in a compact config-defaults list."""

    key: str
    default_str: str
    default_block_str: str | None = None
    type_str: str | None = None
    description: str | None = None
    children: list["ConfigDefaultEntry"] | None = None
    classes: tuple[str, ...] = ()
    anchor_id: str | None = None
    ref_label: str | None = None
    anchor_scope: str | None = None


def _save_field(result: dict[str, str], name: str | None, desc: list[str]) -> None:
    """Save a parsed field into the result dict."""
    if name is not None and desc:
        head, *tail = desc
        body = textwrap.dedent("\n".join(tail))
        text = f"{head}\n{body}".strip() if body else head.strip()
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
            current_desc = [match.group(2)]
            continue

        if current_name is None:
            continue

        if line.startswith(":") or (line and not line[0].isspace()):
            _save_field(result, current_name, current_desc)
            current_name = None
            current_desc = []
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


def _class_summary(cls: Any) -> str | None:
    """Extract the leading summary paragraph from a class docstring.

    Returns:
        Summary paragraph, or ``None`` if unavailable.
    """
    doc = inspect.getdoc(cls)
    if not doc or doc.startswith(f"{cls.__name__}("):
        return None
    lines: list[str] = []
    for line in doc.splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines) if lines else None


def _format_class_ref(cls: type[Any]) -> str:
    return f":py:obj:`{cls.__name__} <{cls.__module__}.{cls.__qualname__}>`"


def _field_public_name(field: Any) -> str:
    """Return the serialized/public config key for a dataclass field."""
    if isinstance(field.default, FieldInfo) and field.default.alias:
        return field.default.alias
    return field.name


def _field_value(instance: Any | None, field: Any) -> Any:
    """Return a field's value on ``instance`` or its declared default."""
    if instance is not None:
        return getattr(instance, field.name)
    if isinstance(field.default, FieldInfo):
        if field.default.is_required():
            return MISSING
        if field.default.default_factory is not None:
            return materialize_config_default(field.default.default_factory)
        return field.default.default
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return materialize_config_default(field.default_factory)
    return MISSING


def _doc_value(value: Any) -> Any:
    """Convert config defaults into user-facing, serialization-like values.

    Returns:
        A docs-friendly value with dataclasses and enums recursively converted
        into plain dict/list/scalar structures.
    """
    if isinstance(value, enum.Enum):
        return value.value
    if is_configurable_dataclass(value) and not isinstance(value, type):
        return _doc_value(config_to_dict(value))
    if isinstance(value, list):
        return [_doc_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_doc_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _doc_value(v) for k, v in value.items()}
    return value


def _format_value(value: Any) -> str:
    """Format an arbitrary default value as RST markup.

    Returns:
        RST-formatted default value string.
    """
    if value is MISSING:
        return "*(required)*"
    if isinstance(value, enum.Enum):
        fq = _enum_target(value)
        return f":py:attr:`{value.name} <{fq}>`" if fq else f"``{value.name}``"
    return f"``{_doc_value(value)!r}``"


def _format_default_block(value: Any) -> str | None:
    """Return YAML text for complex field defaults that read poorly inline.

    Returns:
        Multiline YAML text for nested list/dict defaults, or ``None`` when the
        field should stay on the compact inline metadata line.
    """
    if value is MISSING:
        return None
    block_value = _doc_value(value)
    if not isinstance(block_value, (list, dict)) or not block_value:
        return None
    yaml_str = dump_yaml(block_value).strip()
    return yaml_str if "\n" in yaml_str else None


def _select_default(value: Any, path: str) -> Any:
    """Select a nested default using public config keys and mapping keys.

    Returns:
        The value named by the public configuration path.
    """
    for part in path.split(".") if path else ():
        if is_dataclass(value) and not isinstance(value, type):
            attr = next(
                (
                    field.name
                    for field in dc_fields(value)
                    if _field_public_name(field) == part
                ),
                part,
            )
            value = getattr(value, attr)
        else:
            value = value[part]
    return value


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


def _slugify(text: str) -> str:
    """Return a lowercase slug for HTML ids."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _anchor_slug(key: str, scope: str | None = None) -> str:
    """Return a stable HTML anchor for a config key."""
    if scope:
        scope_slug = _slugify(scope)
        if "." in key:
            prefix, leaf = key.rsplit(".", 1)
            slug = f"{_slugify(prefix)}-{scope_slug}-{_slugify(leaf)}"
        else:
            slug = f"{_slugify(key)}-{scope_slug}"
    else:
        slug = _slugify(key)
    return f"cfg-{slug}" if slug else "cfg"


class ConfigKeyRole(XRefRole):
    """Global ``cfgkey`` role alias for the ``jaqcfg`` domain."""

    def process_link(
        self,
        env: BuildEnvironment,
        refnode: Element,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        refnode["refdomain"] = "jaqcfg"
        refnode["reftype"] = "cfgkey"
        return super().process_link(env, refnode, has_explicit_title, title, target)


class ConfigKeyDomain(Domain):
    """Domain for searchable configuration keys rendered by ``config-defaults``."""

    name = "jaqcfg"
    label = "Configuration Keys"
    data_version: ClassVar[int] = 1

    object_types: ClassVar[dict[str, ObjType]] = {
        "key": ObjType(_("config key"), "cfgkey"),
    }
    roles: ClassVar[dict[str, Any]] = {
        "cfgkey": XRefRole(innernodeclass=nodes.literal, warn_dangling=True),
    }
    initial_data: ClassVar[dict[str, Any]] = {
        "objects": {},
    }

    def note_config_key(
        self,
        ref_label: str,
        anchor_id: str,
        display_name: str,
        location: Any = None,
    ) -> None:
        """Register a configuration key for cross-references and search."""
        if ref_label in self.data["objects"]:
            docname = self.data["objects"][ref_label][0]
            logger.warning(
                __("duplicate config key description of %s, other instance in %s"),
                ref_label,
                docname,
                location=location,
            )
        self.data["objects"][ref_label] = (
            self.env.current_document.docname,
            anchor_id,
            display_name,
        )

    def clear_doc(self, docname: str) -> None:
        to_remove = [
            key
            for key, (stored_docname, _, _) in self.data["objects"].items()
            if stored_docname == docname
        ]
        for key in to_remove:
            del self.data["objects"][key]

    def merge_domaindata(
        self, docnames: AbstractSet[str], otherdata: dict[str, Any]
    ) -> None:
        for key, data in otherdata["objects"].items():
            if data[0] in docnames:
                self.data["objects"][key] = data

    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        typ: str,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> nodes.reference | None:
        if target not in self.data["objects"]:
            return None
        docname, anchor_id, _display_name = self.data["objects"][target]
        return make_refnode(builder, fromdocname, docname, anchor_id, contnode)

    def resolve_any_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> list[tuple[str, nodes.reference]]:
        result = self.resolve_xref(
            env, fromdocname, builder, "cfgkey", target, node, contnode
        )
        if result is None:
            return []
        return [("jaqcfg:cfgkey", result)]

    def get_type_name(self, type: ObjType, primary: bool = False) -> str:
        return type.lname

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        for ref_label, (docname, anchor_id, display_name) in sorted(
            self.data["objects"].items()
        ):
            yield (
                ref_label,
                display_name,
                "key",
                docname,
                anchor_id,
                self.object_types["key"].attrs["searchprio"],
            )


def _resolve_qualified(name: str) -> Any:
    """Resolve a dotted ``module.attr`` path.

    Returns:
        The resolved object.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    module_name, _, attr = name.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Cannot resolve {name}: expected 'module.object'")
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot resolve {name}: {exc}") from exc


class ConfigDefaults(SphinxDirective):
    """Render effective contextual defaults as compact stacked rows."""

    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec: ClassVar[_OptionSpec] = {
        "prefix": directives.unchanged,
        "scope": directives.unchanged,
        "defaults-from": directives.unchanged,
    }

    def _report_error(self, message: str) -> list[nodes.Node]:
        error = self.state_machine.reporter.error(
            message,
            nodes.literal_block(self.block_text, self.block_text),
            line=self.lineno,
        )
        return [error]

    def _resolve_render_target(
        self, prefix: str
    ) -> tuple[type[Any], Any | None] | list[nodes.Node]:
        """Resolve the dataclass type and default instance rendered by this directive.

        Returns:
            The target dataclass type and its default instance, or an error node.
        """
        qualified = self.arguments[0] if self.arguments else None
        source_name = self.options.get("defaults-from")
        if qualified is None and source_name is None:
            return self._report_error(
                "config-defaults requires a class argument or :defaults-from:"
            )
        if qualified is not None and source_name is not None:
            return self._report_error(
                "config-defaults cannot combine a class argument with :defaults-from:"
            )

        try:
            if source_name is None:
                assert qualified is not None
                try:
                    target = _resolve_qualified(qualified)
                except ValueError as exc:
                    return self._report_error(f"Cannot import {qualified}: {exc}")
                if not (isinstance(target, type) and is_dataclass(target)):
                    return self._report_error(f"{qualified} is not a dataclass")
                try:
                    instance = materialize_config_default(target)
                except Exception:
                    instance = None
                return target, instance

            source = _resolve_qualified(source_name)
            if not (isinstance(source, type) and is_dataclass(source)):
                return self._report_error(
                    f"Cannot resolve defaults for {source_name}: "
                    "source is not a dataclass type"
                )
            selected = _select_default(materialize_config_default(source), prefix)
            if not is_dataclass(selected) or isinstance(selected, type):
                return self._report_error(
                    f"Cannot resolve defaults for {source_name}:{prefix}: "
                    "selected value is not a dataclass"
                )
            return type(selected), selected
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            label = qualified or f"{source_name}:{prefix}"
            return self._report_error(f"Cannot resolve defaults for {label}: {exc}")

    def _module_group_description(
        self,
        description: str | None,
        field_key: str,
        module_str: str | None,
    ) -> str:
        """Build the explanatory line for a swappable component block.

        Returns:
            Compact inline description for the group header.
        """
        parts: list[str] = []
        if description:
            parts.append(description.rstrip("."))
        if module_str is None:
            parts.append(
                f"Set ``{field_key}.module`` to switch to a configurable component."
            )
        else:
            parts.append(
                f"Swappable component; the nested keys below are the "
                f"options for the current module {module_str} and change when "
                f"``{field_key}.module`` changes."
            )
        return ". ".join(parts)

    def _format_type(self, field_type: Any) -> str:
        if isinstance(field_type, TypeAliasType):
            field_type = field_type.__value__
        return format_annotation(field_type, self.config)

    def _meta_text(self, entry: ConfigDefaultEntry) -> str:
        """Build the compact metadata line shown under the key.

        Returns:
            Inline metadata string containing default and type.
        """
        parts: list[str] = []
        if entry.default_block_str is None and entry.default_str:
            parts.append(f"Default: {entry.default_str}")
        if entry.type_str:
            parts.append(f"Type: {entry.type_str}")
        return " · ".join(parts)

    def _field_description(
        self, descriptions: dict[str, str], field: Any, mod: object
    ) -> str | None:
        """Return a field description, preferring the public serialized name."""
        public_name = _field_public_name(field)
        text = descriptions.get(public_name, descriptions.get(field.name))
        return _resolve_refs(text, mod) if text else None

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
            cfg_domain = cast(ConfigKeyDomain, self.env.domains["jaqcfg"])
            cfg_domain.note_config_key(
                entry.ref_label,
                entry.anchor_id,
                (
                    f"{entry.key} ({entry.anchor_scope})"
                    if entry.anchor_scope
                    else entry.key
                ),
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
        if len(meta.children) > 0:
            item += meta
        messages.extend(meta_messages)

        if entry.default_block_str is not None:
            default_label = nodes.paragraph(
                classes=["config-defaults-meta", "config-defaults-default-label"]
            )
            label_nodes, label_messages = self.state.inline_text(
                "Default:", self.lineno
            )
            default_label.extend(label_nodes)
            item += default_label
            messages.extend(label_messages)

            default_block = nodes.literal_block(
                entry.default_block_str, entry.default_block_str
            )
            default_block["language"] = "yaml"
            default_block["classes"].append("config-defaults-default-block")
            item += default_block

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
                    replace(
                        child,
                        classes=(*child.classes, "config-defaults-item-nested"),
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
            entry.ref_label = f"{docname.replace('/', '-')}-{entry.anchor_id}"
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
        field: Any,
        field_key: str,
        value: Any,
        descriptions: dict[str, str],
        mod: object,
        anchor_scope: str | None,
    ) -> ConfigDefaultEntry:
        """Collect a grouped entry for a swappable ``ModuleConfig`` field.

        Returns:
            Parent entry with nested current-module options.
        """
        nested_items: list[ConfigDefaultEntry] = []
        has_module_default = is_dataclass(value)
        if has_module_default:
            nested_cls = value if isinstance(value, type) else type(value)
            nested_instance = None if isinstance(value, type) else value
            module_default_str = _format_class_ref(nested_cls)
            nested_items.append(
                ConfigDefaultEntry(
                    key=f"{field_key}.module",
                    default_str=module_default_str,
                    type_str="module path",
                    description="Select the implementation used for this component.",
                    anchor_scope=anchor_scope,
                )
            )
            nested_items.extend(
                self._collect_dataclass_items(
                    nested_cls,
                    field_key,
                    nested_instance,
                    anchor_scope=anchor_scope,
                )
            )
        else:
            module_default_str = _format_value(value)
        return ConfigDefaultEntry(
            key=field_key,
            default_str=module_default_str,
            type_str=self._format_type(field.type),
            description=self._module_group_description(
                self._field_description(descriptions, field, mod),
                field_key,
                module_default_str if has_module_default else None,
            ),
            children=nested_items or None,
            classes=("config-defaults-item-group",) if nested_items else (),
            anchor_scope=anchor_scope,
        )

    def _collect_estimator_collection_items(
        self,
        cls: type[EstimatorCollection],
        prefix: str,
        instance: Any | None,
        anchor_scope: str | None = None,
    ) -> list[ConfigDefaultEntry]:
        items: list[ConfigDefaultEntry] = []
        mod = importlib.import_module(cls.__module__)
        descriptions = _collect_dataclass_descriptions(cls)
        enabled_defaults = getattr(instance, "_enabled", {})
        for field in dc_fields(cls):
            tp = (
                get_args(field.type)[0]
                if get_origin(field.type) is Annotated
                else field.type
            )
            field_cls = get_origin(tp) or tp
            if not isinstance(field_cls, type):
                continue
            name = _field_public_name(field)
            field_key = f"{prefix}.{name}" if prefix else name
            child_val = _field_value(instance, field)
            child_instance = (
                child_val
                if is_dataclass(child_val) and not isinstance(child_val, type)
                else None
            )
            child_mod = importlib.import_module(field_cls.__module__)
            desc = self._field_description(descriptions, field, mod) or _resolve_refs(
                _class_summary(field_cls) or "", child_mod
            )
            is_togglable = field.name in enabled_defaults
            en_key = f"{field_key}.enabled"
            en_val = enabled_defaults[field.name] if is_togglable else None

            if issubclass(field_cls, EstimatorCollection):
                if is_togglable:
                    items.append(
                        ConfigDefaultEntry(
                            key=en_key,
                            default_str=_format_value(en_val),
                            type_str="bool",
                            description=desc or None,
                            anchor_scope=anchor_scope,
                        )
                    )
                items.extend(
                    self._collect_estimator_collection_items(
                        field_cls, field_key, child_instance, anchor_scope
                    )
                )
                continue

            cls_ref = _format_class_ref(field_cls)
            en_str = (
                "always"
                if not is_togglable
                else f"{_format_value(en_val)} (``{en_key}``)"
            )
            children = self._collect_dataclass_items(
                field_cls, field_key, child_instance, anchor_scope
            )
            items.append(
                ConfigDefaultEntry(
                    key=field_key,
                    default_str="",
                    type_str=f"{cls_ref} · Enabled: {en_str}",
                    description=desc or None,
                    children=children or None,
                    classes=("config-defaults-item-group",) if children else (),
                    anchor_scope=anchor_scope,
                )
            )
        return items

    def _collect_dataclass_items(
        self,
        cls: type[Any],
        prefix: str,
        instance: Any | None,
        anchor_scope: str | None = None,
    ) -> list[ConfigDefaultEntry]:
        if issubclass(cls, EstimatorCollection):
            return self._collect_estimator_collection_items(
                cls, prefix, instance, anchor_scope=anchor_scope
            )
        items: list[ConfigDefaultEntry] = []
        mod = importlib.import_module(cls.__module__)
        descriptions = _collect_dataclass_descriptions(cls)
        for field in dc_fields(cls):
            if field.metadata.get("runtime") or field.metadata.get("exclude"):
                continue
            if (
                issubclass(cls, Estimator)
                and field.name == "loss_weight"
                and instance is not None
                and getattr(instance, "loss_weight", None) is None
            ):
                continue

            public_name = _field_public_name(field)
            value = _field_value(instance, field)
            field_key = f"{prefix}.{public_name}" if prefix else public_name

            if _has_module_config(field.type):
                items.append(
                    self._collect_module_config_item(
                        field,
                        field_key,
                        value,
                        descriptions,
                        mod,
                        anchor_scope,
                    )
                )
                continue

            field_type = (
                get_args(field.type)[0]
                if get_origin(field.type) is Annotated
                else field.type
            )
            if isinstance(field_type, type) and is_dataclass(field_type):
                nested_instance = (
                    value
                    if is_dataclass(value) and not isinstance(value, type)
                    else None
                )
                items.extend(
                    self._collect_dataclass_items(
                        field_type,
                        field_key,
                        nested_instance,
                        anchor_scope=anchor_scope,
                    )
                )
                continue

            items.append(
                ConfigDefaultEntry(
                    key=field_key,
                    default_str=_format_value(value),
                    default_block_str=_format_default_block(value),
                    type_str=self._format_type(field.type),
                    description=self._field_description(descriptions, field, mod),
                    anchor_scope=anchor_scope,
                )
            )
        return items

    def _warn_duplicate_scope(self, prefix: str, scope: str | None) -> None:
        docname = self.env.current_document.docname
        scopes_by_doc = self.env.temp_data.setdefault("config-defaults-scopes", {})
        doc_scopes = scopes_by_doc.setdefault(docname, {})
        scope_key = _slugify(scope) if scope else ""
        entry_key = (prefix, scope_key)
        if entry_key in doc_scopes:
            logger.warning(
                __(
                    "duplicate config-defaults scope on %s: prefix=%r scope=%r "
                    "(also at line %s)"
                ),
                docname,
                prefix,
                scope,
                doc_scopes[entry_key],
                location=self.get_location(),
            )
        else:
            doc_scopes[entry_key] = str(self.lineno)

    def run(self) -> list[nodes.Node]:
        self.env.note_dependency(__file__)
        prefix = self.options.get("prefix", "")
        scope = self.options.get("scope")
        if scope is not None:
            scope = scope.strip() or None
        self._warn_duplicate_scope(prefix, scope)
        resolved = self._resolve_render_target(prefix)
        if isinstance(resolved, list):
            return resolved
        target, instance = resolved
        lines = self._collect_dataclass_items(
            target, prefix, instance, anchor_scope=scope
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
    app.add_domain(ConfigKeyDomain)
    app.add_role(
        "cfgkey",
        ConfigKeyRole(innernodeclass=nodes.literal, warn_dangling=True),
        override=True,
    )
    app.add_directive("config-defaults", ConfigDefaults)
    return {"version": "0.1", "parallel_read_safe": True}
