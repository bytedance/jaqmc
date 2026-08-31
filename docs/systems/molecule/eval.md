# Molecule Evaluation

Configuration reference for `jaqmc molecule evaluate`.
This page shows the effective defaults for the evaluation workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Evaluation keys
for `run.*`, `sampler.*`, and `writers.*` live at the config root. Defaults are
resolved in this order: schema defaults, workflow preset, YAML config, then CLI
overrides. For training config, see <project:train.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control evaluation-wide settings and checkpoint loading.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

Must match the training run. The effective defaults and built-in module choices
are identical to the [training system config](#molecule-train-system).

## Wavefunction (`wf.*`)

Must match the training run. The effective defaults and built-in module choices
are identical to the [training wavefunction config](#molecule-train-wf).

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

Molecule evaluation uses adaptive Metropolis-Hastings sampling with an
all-electron Gaussian proposal.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: sampler
```

## Writers (`writers.*`)

The evaluation HDF5 writer is always enabled because its per-step statistics
are required for digest computation. It writes to `evaluation_stats.h5`; other
root-level writer keys enable additional outputs.

### Console writer (`writers.console.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.console.ConsoleWriter
   :prefix: writers.console
```

### CSV writer (`writers.csv.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.csv.CSVWriter
   :prefix: writers.csv
```

(molecule-estimators)=
## Estimators (`estimators.*`)

Energy estimator definitions match training, with additional evaluation-only
estimators enabled through boolean flags.

- `total_energy` and the electron-nuclei potential are always added by the
  workflow and are not configurable via config keys.
- `estimators.enabled.energy` defaults to `true`.
- `estimators.enabled.spin` defaults to `false`.
- `estimators.enabled.density` defaults to `false`.

### Kinetic energy (`estimators.energy.kinetic.*`)

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.EuclideanKinetic
   :prefix: estimators.energy.kinetic
```

### ECP energy (`estimators.energy.ecp.*`)

Added automatically when `system.pp` selects an ECP for at least one atom.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.ecp.estimator.ECPEnergy
   :prefix: estimators.energy.ecp
```

### PH energy (`estimators.energy.ph.*`)

Added automatically when `system.pp` selects PH (`"ph"`) for at least one
atom. PH is the local pseudopotential family, parallel to the semi-local ECP
family, in this workflow: a mixed system may use PH atoms, ECP atoms, and
all-electron atoms together in the same run.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.ph.PHEnergy
   :prefix: estimators.energy.ph
```

### Density (`estimators.density.*`)

Produces a joint histogram of electron positions projected onto selected Cartesian directions.

When enabled without overrides, the workflow wires one 3-D histogram with x, y,
and z as the active axes. Each axis uses 50 bins, and its range is
auto-computed from atom coordinates with 5 bohr padding. To keep only specific
axes, set the others to `null`; one remaining axis gives a 1-D histogram, and
two remaining axes give a 2-D histogram.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.density.cartesian.CartesianAxis
   :prefix: estimators.density.axes.(x|y|z)
```

```yaml
# Just enable with defaults (joint x/y/z histogram):
estimators:
  enabled:
    density: true
```

```yaml
# Keep only z with custom bins and range:
estimators:
  enabled:
    density: true
  density:
    axes:
      x: null
      y: null
      z:
        direction: [0, 0, 1]
        bins: 100
        range: [-15.0, 15.0]
```
