# Training

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

- Default module selection: `mhpo`. Effective defaults for the built-in
  architectures are listed below. Built-in choices are `mhpo`, `laughlin`,
  and `free`.

See <project:index.md> for background on each architecture.

### MHPO options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.mhpo.MHPO
   :prefix: wf
```

### Laughlin options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.laughlin.Laughlin
   :prefix: wf
```

### Free options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.free.Free
   :prefix: wf
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
```

#### SR options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.sr.SROptimizer
   :prefix: train.optim
```

#### Adam options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.adam
   :prefix: train.optim
```

#### LAMB options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.lamb
   :prefix: train.optim
```

(hall-train-sampler)=
### Sampler (`train.sampler.*`)

- Default sampler module: `mcmc`, and its effective keys are listed below.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: train.sampler
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

### Loss gradients

The workflow wires {py:obj}`~jaqmc.estimator.loss_grad.LossAndGrad`
automatically. When angular momentum penalties are enabled
(`system.lz_penalty` or `system.l2_penalty`), the loss key is set to
`penalized_loss`; otherwise it defaults to `total_energy`. There is no
user-facing `train.grads.*` schema for hall workflows.

---

(hall-estimators)=
## Estimators (`estimators.*`)

Energy estimators are configured programmatically by the workflow and are not
typically overridden via config. The same definitions are used by
<project:eval.md>. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

`TotalEnergy` automatically sums all `energy:`-prefixed components. When
`system.lz_penalty` or `system.l2_penalty` are nonzero, a `PenalizedLoss`
estimator is added automatically. Neither is configurable via a config key.

### Kinetic energy (`estimators.energy.kinetic.*`)

Kinetic energy estimator on the Haldane sphere using the covariant Laplacian. See [Kinetic energy](../../guide/estimators/kinetic.md) for physics details and Laplacian mode trade-offs.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Coulomb potential (`estimators.energy.potential.*`)

Coulomb repulsion on the Haldane sphere.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.hamiltonian.SpherePotential
   :prefix: estimators.energy.potential
```
