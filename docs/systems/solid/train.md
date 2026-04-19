# Training

Configuration reference for `jaqmc solid train`.
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
   :preset: jaqmc.app.solid.workflow.SolidTrainWorkflow.default_preset
```

## Workflow (`workflow.*`)

These keys control workflow-level settings shared across all stages.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.base.WorkflowConfig
   :prefix: workflow
```

(solid-train-system)=
## System (`system.*`)

Defines the periodic solid system to simulate. The implementation is selected by
`system.module`.

- Default module selection: unset, so `system.*` is read directly as an
  arbitrary crystal config. Built-in choices are:
  - unset: arbitrary crystal config
  - `rock_salt`: rock-salt crystal generator
  - `two_atom_chain`: two-atom chain generator

### Arbitrary crystals (default)

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.base.SolidConfig
   :prefix: system
```

### Rock salt (`system.module=rock_salt`)

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.rock_salt.rock_salt_config
   :prefix: system
```

### Two-atom chain (`system.module=two_atom_chain`)

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.two_atom_chain.two_atom_chain
   :prefix: system
```

(solid-train-wf)=
## Wavefunction (`wf.*`)

- Default module selection: `solid`. Its effective defaults are listed below.
  The built-in choice is `solid`.

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.wavefunction.SolidWavefunction
   :prefix: wf
```

---

(solid-train-stage)=
## Train Stage (`train.*`)

The main VMC optimization loop. Samples electron configurations, computes energy, and updates wavefunction parameters.

(solid-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
   :prefix: train.run
```

(solid-train-optim)=
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

(solid-train-sampler)=
### Sampler (`train.sampler.*`)

- Default sampler module: `mcmc`, and its effective keys are listed below.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: train.sampler
```

(solid-train-writers)=
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

(solid-train-grads)=
### Loss gradients (`train.grads.*`)

Loss and gradient estimator. Computes the VMC loss and parameter gradients. See [Loss and gradient](../../guide/estimators/loss-grad.md) for the mathematical derivation and outlier clipping details.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.loss_grad.LossAndGrad
   :prefix: train.grads
```

---

## Pretrain Stage (`pretrain.*`)

Initializes the neural network to approximate Hartree-Fock orbitals before VMC
training. It uses the same run, sampler, and writer schemas as the train stage,
but with a different optimizer default and a workflow-wired supervised loss.

### Run options (`pretrain.run.*`)

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
   :prefix: pretrain.run
```

### Optimizer (`pretrain.optim.*`)

- Default optimizer module: `optax:adam`, and its effective keys are listed
  below.

#### Effective Adam defaults

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.adam
   :prefix: pretrain.optim
```

### Sampler (`pretrain.sampler.*`)

- Default sampler module: `mcmc`.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: pretrain.sampler
```

### Writers (`pretrain.writers.*`)

The pretrain stage enables `console`, `csv`, and `hdf5` writers by default.

#### Console writer (`pretrain.writers.console.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.console.ConsoleWriter
   :prefix: pretrain.writers.console
```

#### CSV writer (`pretrain.writers.csv.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.csv.CSVWriter
   :prefix: pretrain.writers.csv
```

#### HDF5 writer (`pretrain.writers.hdf5.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.hdf5.HDF5Writer
   :prefix: pretrain.writers.hdf5
```

### Loss gradients

Pretraining does not use configurable `pretrain.grads.*` settings. The workflow
wires a supervised Hartree-Fock orbital-matching loss directly.

---

(solid-estimators)=
## Estimators (`estimators.*`)

Energy estimators are configured programmatically by the workflow and are not
typically overridden via config. The same definitions are used by
<project:eval.md>. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

`PotentialEnergy` uses [Ewald summation](../../guide/estimators/ewald.md) for
periodic Coulomb interactions and is always present. `TotalEnergy`
automatically sums all `energy:`-prefixed components. Neither is configurable
via a config key.

- `estimators.enabled.spin` defaults to `false`.

### Kinetic energy (`estimators.energy.kinetic.*`)

Kinetic energy estimator. See [Kinetic energy](../../guide/estimators/kinetic.md) for physics details and Laplacian mode trade-offs.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.EuclideanKinetic
   :prefix: estimators.energy.kinetic
```

### ECP energy (`estimators.energy.ecp.*`)

ECP (pseudopotential) energy estimator. Added automatically when `ecp` is set in the system config. See [Pseudopotentials](../../guide/estimators/ecp.md) for physics details and quadrature options.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.ecp.estimator.ECPEnergy
   :prefix: estimators.energy.ecp
```
