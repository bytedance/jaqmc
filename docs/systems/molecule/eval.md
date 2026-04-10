# Evaluation

Configuration reference for `jaqmc molecule evaluate`.
This page shows the effective defaults for the evaluation workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Evaluation has
only one stage, so stage keys (`run.*`, `sampler.*`, `writers.*`) live at the
config root rather than under a `train.*` prefix. Defaults are resolved in this
order: schema defaults, workflow preset, YAML config, then CLI overrides. For
training config, see <project:train.md>.

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
are identical to the
[training wavefunction config](#molecule-train-wf).

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

- Default sampler module: `mcmc`, and its effective keys are listed below.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: sampler
```

## Writers (`writers.*`)

No external writers are enabled by default. If you enable them manually, the
root-level writer keys below control their configuration. The evaluation stage
always writes per-step statistics to an internal HDF5 file for digest
computation; this is independent of the writers configured here.

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

### HDF5 writer (`writers.hdf5.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.hdf5.HDF5Writer
   :prefix: writers.hdf5
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

Added automatically when `system.ecp` is set.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.ecp.estimator.ECPEnergy
   :prefix: estimators.energy.ecp
```

### Density (`estimators.density.*`)

Produces independent 1-D histograms of electron positions projected onto Cartesian directions.

When enabled without overrides, the workflow wires three independent 1-D
histograms along x, y, and z, each with 50 bins and a range auto-computed from
atom coordinates with 5 bohr padding. To keep only specific axes, set the
others to `null`. Each axis override accepts `direction`, `bins`, and `range`.

```yaml
# Just enable with defaults (x, y, z histograms, auto-ranged):
estimators:
  enabled:
    density: true

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
