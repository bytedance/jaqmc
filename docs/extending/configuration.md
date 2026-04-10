# Configuration System

This page documents the `ConfigManager` API for developers building on top of JaQMC. For end-user configuration (CLI overrides, YAML files, `--dry-run`), see <project:../guide/configuration.md>.

## Motivation

JaQMC provides convenient workflow building blocks like the `VMCWorkStage` builder (see <project:writing-workflows.md>) that configure the optimizer, sampler, and estimators with sensible defaults. However, as your research becomes more advanced, you might need to:

- Introduce new custom modules (e.g., a novel optimizer or sampler).
- Change how components are wired together.
- Add new configurable parameters to your custom workflow.

`ConfigManager` is the tool that enables the stage builder to be so flexible, and learning it allows you to write workflows that are just as powerful.

For writing custom configurable components with {deco}`~jaqmc.utils.config.configurable_dataclass`, {func}`~jaqmc.utils.wiring.runtime_dep`, and {func}`~jaqmc.utils.wiring.wire`, see <project:custom-components/index.md>.

## Basic Retrieval (`get`)

Use `cfg.get(key, default)` to retrieve values. The `default` argument serves two purposes: providing a fallback value and **defining the expected type (schema)**.

### Primitives and Containers

For simple types (int, float, str, list, dict), `ConfigManager` ensures the returned value matches the type of the default.

```python
# Returns an int. If config has "train.run.iterations", it is cast to int.
iterations = cfg.get("train.run.iterations", 100)

# Returns a list.
hidden_sizes = cfg.get("wf.hidden_sizes", [64, 64])
```

### Dataclasses

Dataclasses are the recommended way to group related configuration parameters. They provide strong typing and a clear place for documentation.

```python
@dataclass
class RunConfig:
    """Run configuration.

    Args:
        iterations: Total training steps.
        batch_size: Number of walkers.
    """
    iterations: int = 1000
    batch_size: int = 4096

# Automatically populates RunConfig fields from the "train.run" section
run_config = cfg.get("train.run", RunConfig())
```

### Callables

```{note}
**Dataclasses are the recommended approach** for configurable components (shown above). Callables are supported for simpler one-off cases where a full dataclass would be overkill.
```

You can also use a function as a schema. `ConfigManager` will extract arguments from the config that match the function's signature.

```python
def make_optimizer(learning_rate: float = 1e-3, beta1: float = 0.9):
    ...

# partial_opt is a partial function with arguments pre-filled from config
partial_opt = cfg.get("train.optim", make_optimizer)
optimizer = partial_opt()
```

**Ignoring extra arguments**: When `ConfigManager` wraps a callable, it typically returns a helper that **ignores extra keyword arguments** if the target function doesn't accept them. This is extremely useful when passing a shared `context` dictionary (e.g., containing `wavefunction`, `batch_size`) to multiple components where each only needs a subset of the data.

```python
# make_optimizer only takes 'learning_rate', but we pass 'wavefunction' in context.
# The wrapper ensures 'wavefunction' is safely ignored.
optimizer = partial_opt(wavefunction=my_wavefunction)
```

## Dynamic Module Loading (`get_module`)

This is a powerful feature that allows users to swap out entire implementations via config.

```python
from jaqmc.utils.wiring import wire

# get_module returns a dataclass instance when the module is a dataclass
sampler = cfg.get_module("train.sampler", "jaqmc.sampler.mcmc:MCMCSampler")
wire(sampler, **context)  # inject runtime dependencies
```

In the YAML config, the user can specify:

```yaml
train:
  sampler:
    module: my_custom_package.samplers:AdvancedSampler
    step_size: 0.5  # Arguments for AdvancedSampler
```

`ConfigManager` will:
1.  Look for `train.sampler.module`.
2.  Import the specified Python object (class or function).
3.  Use that object as the schema to resolve the rest of the `train.sampler` section.
4.  If the resolved object is a **dataclass**, return an instance with fields populated from config.
5.  If the resolved object is a **callable**, return a partial with config values baked in.

## Nested Modules (`module_config`)

Sometimes a module needs to depend on another configurable module. For example, an optimizer might need a learning rate schedule, or a wavefunction might need a feature builder.

Use `module_config()` as a dataclass field descriptor to express this dependency.

### Code Example

```python
from jaqmc.utils.config import configurable_dataclass, module_config

@configurable_dataclass
class MyOptimizer:
    learning_rate: Any = module_config(LinearDecay, start_value=1.0)
```

(nested-config-syntax)=
### Configuration Behavior

When `ConfigManager` encounters a `module_config` field:
1.  It automatically adds a `learning_rate` section to the configuration.
2.  It sets the default `module` to the path of `LinearDecay`.
3.  It sets other default values (like `start_value=1.0`).
4.  Upon deserialization, it recursively resolves and instantiates the `learning_rate` dataclass before creating `MyOptimizer`.

In YAML, the user can then override the schedule:

```yaml
train:
  optim:
    module: my_lib.optimizers:MyOptimizer
    learning_rate:
      module: my_lib.schedules:CosineDecay  # Swap implementation
      decay_steps: 1000
```

## Collections (`get_collection`)

`get_collection` allows you to retrieve a dynamic set of named modules. This is useful for plugins like writers or estimators where the user might want to enable/disable specific ones or add their own.

When a `context` dict is provided, `get_collection` automatically calls `wire()` on each dataclass instance it creates, injecting runtime dependencies from `context`.

```python
writers = cfg.get_collection(
    "train.writers",
    defaults={
        "console": "jaqmc.writer.console:ConsoleWriter",
        "hdf5": "jaqmc.writer.hdf5:HDF5Writer",
    },
    context=context,  # auto-wires runtime deps into each writer
)
```

The user can disable a default writer by setting it to `null` in YAML, or add a new one:

```yaml
train:
  writers:
    console: null  # Disable console output
    my_logger:     # Add custom logger
      module: my_code:MyLogger
      log_dir: /tmp/logs
```

## Finalization

After the application has initialized all its components, you should call `cfg.finalize()`.

```python
cfg.finalize()
```

This method:
1.  Logs the fully resolved configuration.
2.  **Checks for unused keys**: If the user provided configuration keys (in YAML or CLI) that were never accessed via `get` or `get_module`, `finalize` will print a warning and (by default) raise an error. This catches typos like `train.run.iteration` instead of `train.run.iterations`.
