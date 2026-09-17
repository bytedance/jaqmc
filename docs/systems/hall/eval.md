# Quantum Hall Evaluation

Configuration reference for `jaqmc hall evaluate`.
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

`workflow.source_path` selects the trained parameters and sampler state to
evaluate. It is required for MHPO and other trainable wavefunctions. The
analytic `laughlin` and `free` wavefunctions have no parameters, so they can
start evaluation from fresh walkers without a source path.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

When evaluating a training checkpoint, these settings must match the training
run. For direct `laughlin` or `free` evaluation, they define the target system.
The effective defaults are identical to the [training system config](#hall-train-system).

## Wavefunction (`wf.*`)

When evaluating a training checkpoint, these settings must match the
training run. For direct `laughlin` or `free` evaluation, select the
analytic module with `wf.module`. MHPO options match the
[training wavefunction config](#hall-train-wf).

### Laughlin options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.laughlin.Laughlin
   :prefix: wf
   :scope: Laughlin
```

### Free options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.free.Free
   :prefix: wf
   :scope: Free
```

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

Hall evaluation uses adaptive Metropolis-Hastings sampling with a spherical
proposal.

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

(hall-estimators)=
## Estimators (`estimators.*`)

Energy estimator definitions match training, with additional evaluation-only
estimators enabled through boolean flags.

- `TotalEnergy` is added automatically by the workflow and is not configurable
  via a config key.
- When `system.lz_penalty` or `system.l2_penalty` are nonzero, a
  `PenalizedLoss` estimator is added automatically. That requires both energy
  and angular momentum to be enabled. Reported energy remains `total_energy`.
- `estimators.enabled.energy` defaults to `true`.
- `estimators.enabled.angular_momentum` defaults to `true`. Setting it to
  `false` while an angular-momentum penalty is active raises a configuration
  error.
- `estimators.enabled.density` defaults to `false`.
- `estimators.enabled.pair_correlation` defaults to `false`.
- `estimators.enabled.one_rdm` defaults to `false`.

### Kinetic energy (`estimators.energy.kinetic.*`)

Covariant kinetic energy on the Haldane sphere. See
[Kinetic energy](../../guide/estimators/kinetic.md#spherical-kinetic-energy).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Angular momentum (`estimators.angular_momentum.*`)

Computes `angular_momentum_z`, `angular_momentum_z_square`, and
`angular_momentum_square` on the Haldane sphere. See
[Angular momentum](../../guide/estimators/angular-momentum.md).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.angular_momentum.SphericalAngularMomentum
   :prefix: estimators.angular_momentum
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

Computes the pair correlation function $g(\theta)$ on the Haldane sphere
from geodesic pair angles, weighted by $1/\sin\theta$. Divide the
accumulated state by the evaluation step count to get the final $g(\theta)$.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.pair_correlation.PairCorrelation
   :prefix: estimators.pair_correlation
```

### One-body RDM (`estimators.one_rdm.*`)

Computes the one-body reduced density matrix in the monopole harmonic
basis. The trace is the number of electrons on the lowest Landau level,
$N_\text{LLL}$.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.one_rdm.OneRDM
   :prefix: estimators.one_rdm
```
