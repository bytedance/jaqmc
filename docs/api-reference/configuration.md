# Configuration

{func}`~jaqmc.utils.config.parse_config` merges YAML layers and CLI
overrides into a typed dataclass. Workflows expose the same operation through
{meth}`~jaqmc.workflow.base.Workflow.from_config`.

For a conceptual overview of the config system, see <project:../guide/configuration.md> in the user guide and <project:../extending/configuration.md> in the extending guide.

## Configuration API

```{eval-rst}
.. autofunction:: jaqmc.utils.config.parse_config

.. autofunction:: jaqmc.utils.config.config_to_yaml

.. autoclass:: jaqmc.utils.config.ModuleConfig
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

```
