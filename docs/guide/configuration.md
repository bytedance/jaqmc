# Configuring Simulations

Every JaQMC command has a structured set of options with built-in defaults, and unknown keys or invalid types are rejected before a run starts. Simulations can be configured through YAML files, command-line overrides, or a combination of both.

## Inspecting Configuration

The `--dry-run` flag validates the configuration and prints the resolved YAML without running SCF calculations or executing the workflow:

```bash
jaqmc molecule train --dry-run
```

Each subcommand (`train` or `evaluate`) defines its own configuration tree, and available keys depend on which component implementations are selected. Running `--dry-run` on a command shows every active section, key, and default value:

```yaml
workflow:
  seed: null
  batch_size: 4096
sampler:
  module: jaqmc.sampler.mcmc:MCMCSampler
  steps: 10
train:
  run:
    iterations: 200000
  optim:
    module: jaqmc.optimizer.kfac:KFACOptimizer
    learning_rate:
      module: jaqmc.optimizer.schedule:Standard
      rate: 0.05
```

## CLI Overrides

Any parameter in the configuration tree can be overridden on the command line using dot-notation that matches its nested YAML path:

```bash
# Override learning rate, training iterations, and sampler steps
jaqmc molecule train \
  train.optim.learning_rate.rate=0.01 \
  train.run.iterations=500 \
  sampler.steps=20
```

Overrides take the form `key=value` without spaces around `=`. Each value is parsed as YAML, so numbers, booleans, lists, and `null` keep their types.

## YAML Files

YAML files passed with `--yml` (or `--yaml`) group settings for larger experiments and reproducible runs:

```yaml
# base.yml
workflow:
  batch_size: 2048
sampler:
  steps: 20
train:
  run:
    iterations: 500
  optim:
    learning_rate:
      rate: 0.01
```

Command-line overrides are applied after all YAML files and take precedence:

```bash
jaqmc molecule train --yml base.yml --yml experiments/li_atom.yml train.run.iterations=1000
```

Multiple files are merged from left to right. When the same key appears in more than one file, the rightmost file wins.

(swappable-modules)=
## Swappable Modules

Many JaQMC components, including wavefunctions (`wf`), optimizers (`train.optim`), samplers (`sampler`), learning-rate schedules (`train.optim.learning_rate`), and writers (`train.writers.*`), can be swapped by setting a `module` key.

Setting fields on a component without changing `module` updates the currently selected implementation. Setting `module` selects a different implementation, which can be configured in the same YAML mapping or on the command line:

```yaml
train:
  optim:
    module: optax:adam
    learning_rate:
      rate: 0.001
```

```bash
# Equivalent command-line override
jaqmc molecule train train.optim.module=optax:adam train.optim.learning_rate.rate=0.001
```

### Module path syntax

A module path identifies the Python object to import. The full form separates the module and object name with a colon:

```text
package.subpackage.module:name
```

Here `name` is the class to import from the module. The module part can also be a path to a `.py` file, such as `path/to/custom.py:MyWavefunction`.

When a Python module defines `__all__`, JaQMC treats the first listed name (`__all__[0]`) as that module's default object, so the `:name` suffix can be omitted:

```text
package.subpackage.module
```

For example, `jaqmc.optimizer.kfac` sets `__all__ = ["KFACOptimizer"]`, so these two paths resolve to the same class:

```bash
train.optim.module=jaqmc.optimizer.kfac:KFACOptimizer
train.optim.module=jaqmc.optimizer.kfac
```

Modules that do not define `__all__`, such as `jaqmc.optimizer.optax`, require an explicit `:name` (`jaqmc.optimizer.optax:adam`).

### Relative resolution

JaQMC first tries to resolve a `module` path relative to the field's default package. If no matching module exists there, it falls back to importing the path as a full module path.

For example, the training optimizer field resolves relative paths against `jaqmc.optimizer`. Setting `train.optim.module=optax:adam` first checks `jaqmc.optimizer.optax:adam`. Because that module exists, it resolves there.

Short names therefore work for built-in alternatives, while full module paths or `.py` file paths work for custom packages:

```bash
# Short name resolved relative to jaqmc.optimizer
train.optim.module=optax:adam

# Full module path for an external package
train.optim.module=my_project.optimizers:CustomOptimizer
```

(switching-modules-across-files)=
### Switching modules across files and overrides

When combining YAML files and CLI overrides:

- Keeping the same module (by omitting `module` or setting it to the same implementation, whether by short name or full path) merges the new settings with earlier ones.
- Switching `module` to a different implementation discards earlier settings for that component and starts from the new module's defaults.
- The workflow's default module starts from the workflow defaults shown by `--dry-run` (for example, a training workflow customizes `train.writers.console.fields` beyond `ConsoleWriter`'s own default). Any other module starts from its own built-in defaults.

With the following file, these rules determine which settings apply in each command below.

```yaml
# psiformer.yml
wf:
  module: psiformer
  num_layers: 2
```

```bash
# Psiformer with num_layers=2 and num_heads=8
jaqmc molecule train --yml psiformer.yml wf.num_heads=8

# LapNet with its built-in defaults (switching module discards num_layers=2)
jaqmc molecule train --yml psiformer.yml wf.module=lapnet

# LapNet with num_layers=2 (given together with the switch)
jaqmc molecule train --yml psiformer.yml wf.module=lapnet wf.num_layers=2
```

(replacing-module-components)=
### Replacing a component with a plain value or `null`

Some fields accept a plain scalar or `null` in place of a component mapping.

- **Scalar shorthand.** Fields such as `learning_rate` accept a number directly. Setting `train.optim.learning_rate=0.1` replaces the learning-rate schedule object with a constant float, whereas `train.optim.learning_rate.rate=0.1` keeps the schedule module and updates its `rate` field.
- **Disabling or adding named entries.** Named component collections such as `writers` allow individual entries to be disabled by setting them to `null` (`train.writers.npy=null`), while entries not enabled by default can be added by setting their `module` key (`train.writers.wandb.module=wandb`).

A plain value or `null` replaces the component entry entirely, discarding the workflow's customized defaults and any settings from earlier YAML files. If a later YAML file or CLI override selects a `module` for that component again, it starts from the component's built-in defaults rather than the workflow's customized preset. For example, a training workflow customizes `train.writers.console.fields`, so setting `train.writers.console=null` in one file and re-enabling it with `train.writers.console.module=console` in a later file or CLI override starts `ConsoleWriter` with its built-in `fields` unless `fields` is specified again.

## Reference

Root-level runtime keys (`logging.*`, `jax.*`, and `distributed.*`) are shared across all commands. <project:runtime-configuration.md> lists their defaults and behavior.

System-specific configuration keys and defaults are documented on each system page:

- <project:../systems/molecule/index.md>
- <project:../systems/solid/index.md>
- <project:../systems/electron_gas/index.md>
- <project:../systems/moire/index.md>
- <project:../systems/hall/index.md>

Component guides cover the available options for each subsystem:

- <project:wavefunction.md>
- <project:estimators/index.md>
- <project:optimizers/index.md>
- <project:sampling.md>
- <project:writers.md>

For typed schema definitions, programmatic parsing, and custom module fields, see <project:../extending/configuration.md>.
