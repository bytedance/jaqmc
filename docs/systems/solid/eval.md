# Evaluation

Configuration reference for `jaqmc solid evaluate`.
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
are identical to the [training system config](#solid-train-system).

## Wavefunction (`wf.*`)

Must match the training run. The effective defaults and built-in module choices
are identical to the [training wavefunction config](#solid-train-wf).

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

## Estimators (`estimators.*`)

Energy estimator definitions match training, with additional evaluation-only
estimators enabled through boolean flags.

- `PotentialEnergy` and `TotalEnergy` are added automatically by the workflow
  and are not configurable via config keys.
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

Electron density in fractional (lattice) coordinates. Converts Cartesian positions to fractional coordinates via the inverse lattice matrix, then histograms within $[0, 1)$.

When enabled without overrides, the workflow wires three independent 1-D
histograms along the `a`, `b`, and `c` lattice directions, each with 50 bins.
To keep only specific axes, set the others to `null`. Each axis override
accepts `lattice_index` and `bins`.

```yaml
# Just enable with defaults (a, b, c histograms, 50 bins each):
estimators:
  enabled:
    density: true

# Keep only c-axis with finer bins:
estimators:
  enabled:
    density: true
  density:
    axes:
      a: null
      b: null
      c:
        lattice_index: 2
        bins: 100
```
