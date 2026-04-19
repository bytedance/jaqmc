# Runtime Configuration

Runtime configuration keys live at the root of the config file. JaQMC applies
them before the workflow starts, so they are shared by molecule, solid, and
Hall commands instead of belonging to one system reference page.

Use these keys for startup behavior: logging, JAX global flags, and optional
JAX distributed initialization. For workflow, system, optimizer, sampler, and
writer keys, use the command-specific configuration references under each
system page.

## Logging (`logging.*`)

```{eval-rst}
.. config-defaults:: jaqmc.utils.runtime.LoggingConfig
   :prefix: logging
```

## JAX runtime (`jax.*`)

```{eval-rst}
.. config-defaults:: jaqmc.utils.runtime.JaxConfig
   :prefix: jax
```

## Distributed runtime (`distributed.*`)

Use `distributed.*` only for multi-host runs. For launch examples and cluster
setup notes, see <project:multi-device.md>.

```{eval-rst}
.. config-defaults:: jaqmc.utils.runtime.DistributedConfig
   :prefix: distributed
```
