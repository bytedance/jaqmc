# Configuration

{class}`~jaqmc.utils.config.ConfigManager` is the central configuration API. It merges YAML files, CLI overrides, and preset configs into a single resolved config tree. Use {meth}`~jaqmc.utils.config.ConfigManager.get` to read values, {meth}`~jaqmc.utils.config.ConfigManager.get_module` to instantiate swappable components, and {meth}`~jaqmc.utils.config.ConfigManager.get_collection` to build named collections of components.

For a conceptual overview of the config system, see <project:../guide/configuration.md> in the user guide and <project:../extending/configuration.md> in the extending guide.

## Config manager

```{eval-rst}
.. autoclass:: jaqmc.utils.config.ConfigManager
   :members: get, get_module, get_collection, scoped, use_preset, finalize, to_yaml

.. autoclass:: jaqmc.utils.config.ScopedConfigManager
   :members: get, get_module, get_collection
```

## Component wiring

Components use {func}`~jaqmc.utils.wiring.runtime_dep` to declare dependencies that are injected at runtime rather than set in config. The {func}`~jaqmc.utils.wiring.wire` function injects those dependencies into a component instance.

```{eval-rst}
.. autofunction:: jaqmc.utils.config.configurable_dataclass

.. autofunction:: jaqmc.utils.wiring.runtime_dep

.. autofunction:: jaqmc.utils.wiring.wire

.. autofunction:: jaqmc.utils.wiring.check_wired
```

## Module resolution

{func}`~jaqmc.utils.module_resolver.resolve_object` resolves ``"module:name"`` strings into Python objects — the mechanism behind swappable components in YAML config.

```{eval-rst}
.. autofunction:: jaqmc.utils.module_resolver.resolve_object

.. autofunction:: jaqmc.utils.module_resolver.import_module_or_file
```

## YAML formatting

```{eval-rst}
.. autofunction:: jaqmc.utils.yaml_format.dump_yaml

.. autofunction:: jaqmc.utils.yaml_format.annotate_yaml_with_sources
```

## Config-facing enums

```{eval-rst}
.. autoclass:: jaqmc.app.hall.config.InteractionType
```
