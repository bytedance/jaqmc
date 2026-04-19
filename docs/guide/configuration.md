# Configuring Simulations

JaQMC uses a code-driven configuration system. For end-users, configuration is handled via command-line overrides and YAML files.

## CLI Overrides

Override any configuration parameter using dot-notation directly on the command line:

```bash
# Override learning rate and number of iterations
jaqmc molecule train train.optim.learning_rate.rate=0.01 train.run.iterations=500
```

Note: do not put spaces around `=` (use `key=value`, not `key = value`).

## YAML Files

For complex experiments or reproducible runs, define configuration in YAML files and pass them with `--yml`. CLI overrides always take precedence.

```bash
jaqmc molecule train --yml base.yml --yml experiments/li_atom.yml train.run.iterations=1000
```

Multiple files are merged left to right — if the same key appears in more than one file, the rightmost file wins.

## Inspecting Configuration

Because the available config keys depend on which modules you select (e.g., KFAC and Adam expose different parameters), it can be helpful to see what configuration the system is actually using:

- **`--dry-run`**: Loads the configuration and initializes the workflow but stops before the expensive training loop. The full resolved config is printed to the terminal and saved as `config.yaml` in the output directory.
- **Source location comments**: The output includes `# Defined in ...` comments pointing to where each parameter is defined in source code.
- **`workflow.config.verbose=true`**: Also includes docstrings for each configured class, helping you understand what each parameter does.

```bash
jaqmc molecule train --dry-run workflow.config.verbose=true
```

## Common Configuration Patterns

### Changing the optimizer

```bash
# Use Adam instead of KFAC for training
jaqmc molecule train train.optim.module=optax:adam

# Adjust learning rate
jaqmc molecule train train.optim.learning_rate.rate=0.001
```

The `optax:adam` syntax is a module path — see [Swappable Modules](#swappable-modules) below for details.

### Adjusting the sampler

```bash
# More MCMC steps between training steps
jaqmc molecule train train.sampler.steps=20
```

### Customizing writers

```bash
# Customize console output to show more precision
jaqmc molecule train train.writers.console.fields="pmove:.2f,energy=total_energy:.6f,variance=total_energy_var:.6f"
```

### Disabling a component

Set a writer to `null` in YAML to disable it:

```yaml
train:
  writers:
    hdf5: null  # Disable HDF5 output
```

(swappable-modules)=
## Swappable Modules

Many JaQMC components — optimizers, samplers, learning rate schedules, writers — can be swapped at runtime by setting a `module` key, as shown in the optimizer example above. This section explains the module path syntax so you can use built-in alternatives or plug in your own implementations.

### Module path syntax

A module path tells JaQMC which Python object to import. The full form is:

```
package.subpackage.module:name
```

where `name` is the specific class or function to import from the module. If the module has a single primary export, you can omit the name:

```
package.subpackage.module
```

For example, these two are equivalent because `adam` is the primary export of `jaqmc.optimizer.optax`:

```bash
train.optim.module=jaqmc.optimizer.optax:adam
train.optim.module=jaqmc.optimizer.optax
```

### Relative resolution

When you set a `module` key, JaQMC first tries to resolve it relative to the default module's package. If that fails, it falls back to treating the value as an absolute path.

For example, the training optimizer defaults to `jaqmc.optimizer.kfac`. If you set:

```bash
train.optim.module=optax:adam
```

JaQMC first tries `jaqmc.optimizer.optax:adam` (relative to `jaqmc.optimizer`). Since that module exists, it resolves there. If it didn't exist, JaQMC would try importing `optax:adam` as an absolute path.

This means you can use short names for built-in modules and full paths for external ones:

```bash
# Short name — resolved relative to jaqmc.optimizer
train.optim.module=optax:adam

# Full path — works for your own packages
train.optim.module=my_project.optimizers:CustomOptimizer
```

## Reference

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:runtime-configuration.md> for their
defaults and field descriptions.

For the full list of configurable keys for each workflow, see the config reference:

- <project:../systems/molecule/index.md>
- <project:../systems/solid/index.md>
- <project:../systems/hall/index.md>

For background on individual component types, see the guide pages:

- <project:estimators/index.md>
- <project:optimizers/index.md>
- <project:sampling.md>
- <project:writers.md>

For the developer-facing {class}`~jaqmc.utils.config.ConfigManager` API (programmatic usage, {meth}`~jaqmc.utils.config.ConfigManager.get_module`, {meth}`~jaqmc.utils.config.ConfigManager.get_collection`), see <project:../extending/configuration.md>.
