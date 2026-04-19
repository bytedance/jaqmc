# Evaluation

Configuration reference for `jaqmc hall evaluate`.
This page shows the effective defaults for the evaluation workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Evaluation has
only one stage, so stage keys (`run.*`, `sampler.*`, `writers.*`) live at the
config root rather than under a `train.*` prefix. Defaults are resolved in this
order: schema defaults, workflow preset, YAML config, then CLI overrides. For
training config, see <project:train.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control evaluation-wide settings and checkpoint loading.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

Must match the training run. The effective defaults are identical to the
[training system config](#hall-train-system).

## Wavefunction (`wf.*`)

Must match the training run. The effective defaults and built-in module choices
are identical to the [training wavefunction config](#hall-train-wf).

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

- `TotalEnergy` is added automatically by the workflow and is not configurable
  via a config key.
- When `system.lz_penalty` or `system.l2_penalty` are nonzero, a
  `PenalizedLoss` estimator is added automatically.
- `estimators.enabled.energy` defaults to `true`.
- `estimators.enabled.density` defaults to `false`.
- `estimators.enabled.pair_correlation` defaults to `false`.
- `estimators.enabled.one_rdm` defaults to `false`.

### Kinetic energy (`estimators.energy.kinetic.*`)

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Coulomb potential (`estimators.energy.potential.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.hamiltonian.SpherePotential
   :prefix: estimators.energy.potential
```

### Density (`estimators.density.*`)

Accumulates a histogram of the polar angle $\theta$ to measure electron density on the sphere.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.density.spherical.SphericalDensity
   :prefix: estimators.density
```

### Pair correlation (`estimators.pair_correlation.*`)

Computes the pair correlation function $g(\theta)$ on the Haldane sphere.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.pair_correlation.PairCorrelation
   :prefix: estimators.pair_correlation
```

### One-body RDM (`estimators.one_rdm.*`)

Computes the one-body reduced density matrix in the monopole harmonic basis.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.one_rdm.OneRDM
   :prefix: estimators.one_rdm
```
