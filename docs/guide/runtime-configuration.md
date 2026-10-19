# Runtime Configuration

Runtime configuration keys live at the root of the config file. JaQMC applies
them before the workflow starts, so they are shared by molecule, solid, and
Hall commands instead of belonging to one system reference page.

These keys configure logging, JAX global flags, and optional JAX distributed
initialization at startup. For workflow, system, optimizer, sampler, and writer
keys, use the command-specific configuration references under each system page.

## Logging (`logging.*`)

```{eval-rst}
.. config-defaults:: jaqmc.utils.runtime.LoggingConfig
   :prefix: logging
```

## JAX runtime (`jax.*`)

Standard JAX environment variables (such as `JAX_ENABLE_X64=1` or
`JAX_DEBUG_NANS=1`) work as usual. Any `jax.*` key left unset in YAML or on the
CLI inherits the active `jax.config` state at startup, while setting a `jax.*`
key explicitly overrides the environment variable. The one exception to JAX's
stock defaults is `jax.default_matmul_precision`, which JaQMC defaults to
`"float32"` when `JAX_DEFAULT_MATMUL_PRECISION` is unset so GPUs do not silently
drop to reduced-precision TensorFloat-32 (TF32).

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
