# Training

Configuration reference for `jaqmc molecule train`.
This page shows the effective defaults for the train workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Keys use the same
dot notation as CLI overrides, such as `train.run.iterations=5000`. Defaults
are resolved in this order: schema defaults, workflow preset, YAML config, then
CLI overrides. For evaluation config, see <project:eval.md>.

```{eval-rst}
.. config-context::
   :preset: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow.default_preset
```

## Workflow (`workflow.*`)

These keys control workflow-level settings shared across all stages.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.base.WorkflowConfig
   :prefix: workflow
```

(molecule-train-system)=
## System (`system.*`)

Defines the molecular system to simulate. The implementation is selected by
`system.module`.

- Default module selection: unset, so `system.*` is read directly as an arbitrary
  molecule config. Built-in choices are:
  - unset: arbitrary molecule config
  - `atom`: single-atom generator
  - `diatomic`: diatomic generator

### Arbitrary molecules (default)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.config.base.MoleculeConfig
   :prefix: system
```

### Single atoms (`system.module=atom`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.config.atom.atom_config
   :prefix: system
```

### Diatomic molecules (`system.module=diatomic`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.config.diatomic.diatomic_config
   :prefix: system
```

(molecule-train-wf)=
## Wavefunction (`wf.*`)

Selects and configures the neural-network ansatz.

- Default module selection: `ferminet`. Effective defaults for the built-in
  architectures are listed below. Built-in choices are `ferminet` and
  `psiformer`.

### FermiNet options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.wavefunction.ferminet.FermiNetWavefunction
   :prefix: wf
```

### Psiformer options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.wavefunction.psiformer.PsiformerWavefunction
   :prefix: wf
```

---

(train-stage)=
## Train Stage (`train.*`)

The main VMC optimization loop. Samples electron configurations, computes energy, and
updates wavefunction parameters.

(train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
   :prefix: train.run
```

(train-optim)=
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

(train-sampler)=
### Sampler (`train.sampler.*`)

- Default sampler module: `mcmc`, and its effective keys are listed below.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: train.sampler
```

(train-writers)=
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

(train-grads)=
### Loss gradients (`train.grads.*`)

Loss and gradient estimator. Computes the VMC loss and parameter gradients.

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

## Estimators (`estimators.*`)

Energy estimators are configured programmatically by the workflow and are not
typically overridden via config. The same definitions are used by <project:eval.md>.

- `total_energy` and the electron-nuclei potential are always added by the workflow
  and are not configurable via config keys.
- `estimators.enabled.spin` defaults to `false`.

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
