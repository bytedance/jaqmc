# Solid Evaluation

Configuration reference for `jaqmc solid evaluate`.
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
are identical to the [training system config](#solid-train-system).

## Wavefunction (`wf.*`)

Must match the training run. The effective defaults and built-in module choices
are identical to the [training wavefunction config](#solid-train-wf).

## Reference (`reference.*`)

The Hartree-Fock reference is the PySCF calculation JaQMC uses when it needs
reference orbitals or related setup from that calculation. Itis recommended to
set the values to match the reference configuration used during training.

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.base.SolidPretrainReferenceConfig
   :prefix: reference
```

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

Solid evaluation uses adaptive Metropolis-Hastings sampling with a periodic
Gaussian proposal.

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

(solid-estimators)=
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

Added automatically when `system.pp` selects an ECP for at least one atom.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.ecp.estimator.ECPEnergy
   :prefix: estimators.energy.ecp
```

### Density (`estimators.density.*`)

Electron density in fractional (lattice) coordinates. Converts Cartesian positions to fractional coordinates via the inverse lattice matrix, then histograms within $[0, 1)$.

When enabled without overrides, the workflow wires one 3-D histogram with the
`a`, `b`, and `c` fractional coordinates as the active axes. Each axis uses 50
bins over $[0, 1)$. To keep only specific axes, set the others to `null`; one
remaining axis gives a 1-D histogram, and two remaining axes give a 2-D
histogram.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.density.fractional.FractionalAxis
   :prefix: estimators.density.axes.(a|b|c)
```

```yaml
# Just enable with defaults (joint a/b/c histogram, 50 bins each):
estimators:
  enabled:
    density: true
```

```yaml
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
