# Quantum Hall Training

Configuration reference for `jaqmc hall train`.
This page shows the effective defaults for the train workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Keys use the same
dot notation as CLI overrides, such as `train.run.iterations=5000`. Defaults
are resolved in this order: schema defaults, workflow preset, YAML config, then
CLI overrides. For evaluation config, see <project:eval.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

```{eval-rst}
.. config-context::
   :preset: jaqmc.app.hall.workflow.HallTrainWorkflow.default_preset
```

## Workflow (`workflow.*`)

These keys control workflow-level settings shared across all stages.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.base.WorkflowConfig
   :prefix: workflow
```

(hall-train-system)=
## System (`system.*`)

Defines the quantum Hall system on the Haldane sphere.

See <project:index.md> for physics background and usage examples.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.config.HallConfig
   :prefix: system
```

(hall-train-wf)=
## Wavefunction (`wf.*`)

Selects and configures the neural network ansatz.

- Default module selection: `mhpo`. Effective defaults for MHPO are
  listed below.

Analytic `laughlin` and `free` benchmarks are evaluation-only; see
<project:eval.md>.

See <project:index.md> for background on each architecture.

### MHPO options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.mhpo.MHPO
   :prefix: wf
   :scope: MHPO
```

(hall-train-sampler)=
## Sampler (`sampler.*`)

The Hall workflow uses adaptive Metropolis-Hastings sampling with a spherical
proposal.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: sampler
```

(hall-train-stage)=
## Train Stage (`train.*`)

The VMC optimization loop. Samples electron configurations on the Haldane sphere, computes energy (and optional angular momentum penalties), and updates wavefunction parameters.

(hall-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
   :prefix: train.run
```

(hall-train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. Effective defaults for the built-in
  optimizers are listed below.

#### KFAC options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.kfac.kfac.KFACOptimizer
   :prefix: train.optim
   :scope: KFAC
```

#### SR options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.sr.SROptimizer
   :prefix: train.optim
   :scope: SR
```

#### Adam options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.adam
   :prefix: train.optim
   :scope: Adam
```

#### LAMB options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.lamb
   :prefix: train.optim
   :scope: LAMB
```

(hall-train-writers)=
### Writers (`train.writers.*`)

The train stage enables `console`, `csv`, and `hdf5` writers by default.

#### Console writer (`train.writers.console.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.console.ConsoleWriter
   :prefix: train.writers.console
```

#### CSV writer (`train.writers.csv.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.csv.CSVWriter
   :prefix: train.writers.csv
```

#### HDF5 writer (`train.writers.hdf5.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.hdf5.HDF5Writer
   :prefix: train.writers.hdf5
```

### Loss gradients (`train.grads.*`)

The workflow resolves {py:obj}`~jaqmc.estimator.loss_grad.LossAndGrad`
from `train.grads.*`. See [Loss and gradient](../../guide/estimators/loss-grad.md) for the clipping formulas.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.loss_grad.LossAndGrad
   :prefix: train.grads
```

---

(hall-train-estimators)=
## Estimators (`estimators.*`)

Energy estimators are configured programmatically by the workflow and are not
typically overridden via config. The same definitions are used by
<project:eval.md>. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

`TotalEnergy` automatically sums all `energy:`-prefixed components.
`SphericalAngularMomentum` is enabled by default; set
`estimators.enabled.angular_momentum=false` to omit it. When
`system.lz_penalty` or `system.l2_penalty` are nonzero, angular momentum
is required by `PenalizedLoss`. `TotalEnergy` and `PenalizedLoss` are not
configurable via config keys.

### Kinetic energy (`estimators.energy.kinetic.*`)

Covariant kinetic energy on the Haldane sphere. See
[Kinetic energy](../../guide/estimators/kinetic.md#spherical-kinetic-energy).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Angular momentum (`estimators.angular_momentum.*`)

Computes `angular_momentum_z`, `angular_momentum_z_square`, and
`angular_momentum_square` for the Haldane sphere. See
[Angular momentum](../../guide/estimators/angular-momentum.md).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.angular_momentum.SphericalAngularMomentum
   :prefix: estimators.angular_momentum
```

### Coulomb potential (`estimators.energy.potential.*`)

Coulomb repulsion on the Haldane sphere.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.hamiltonian.SpherePotential
   :prefix: estimators.energy.potential
```
