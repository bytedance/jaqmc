# Molecule Training

Configuration reference for `jaqmc molecule train`.
This page shows the train workflow defaults. `--dry-run` prints the
resolved config for a specific run. Keys use the same dot notation as CLI overrides,
such as `train.run.iterations=5000`. Defaults are resolved in this order:
schema defaults, YAML config, then CLI overrides. For evaluation config, see
<project:eval.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control workflow-level settings shared across all stages.

```{eval-rst}
.. config-defaults::
   :prefix: workflow
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

(molecule-train-system)=
## System (`system.*`)

Defines the molecular system to simulate. The implementation is selected by
`system.module`.

- Default module selection: `base:MoleculeConfig`, so `system.*` is read
  directly as an arbitrary molecule config when `system.module` is omitted.
  Built-in choices are:
  - `base:MoleculeConfig` (or omitted): arbitrary molecule config
  - `atom`: single-atom generator
  - `diatomic`: diatomic generator

### Arbitrary molecules (default)

```{eval-rst}
.. config-defaults::
   :prefix: system
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
   :scope: Molecule
```

### Single atoms (`system.module=atom`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.config.atom.AtomBuilder
   :prefix: system
   :scope: atom
```

### Diatomic molecules (`system.module=diatomic`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.config.diatomic.DiatomicBuilder
   :prefix: system
   :scope: diatomic
```

(molecule-train-wf)=
## Wavefunction (`wf.*`)

Selects and configures the neural-network ansatz.

- Default module selection: `ferminet`. Effective defaults for the built-in
  architectures are listed below. Built-in molecule choices are `ferminet`,
  `lapnet`, and `psiformer`.
- For architecture selection, examples, and the main tuning knobs for each
  built-in option, see <project:../../guide/wavefunction.md>.

### FermiNet options (`wf.*`)

```{eval-rst}
.. config-defaults::
   :prefix: wf
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
   :scope: FermiNet
```

### LapNet options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.wavefunction.lapnet.LapNetWavefunction
   :prefix: wf
   :scope: LapNet
```

### Psiformer options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.wavefunction.psiformer.PsiformerWavefunction
   :prefix: wf
   :scope: Psiformer
```

(train-sampler)=
## Sampler (`sampler.*`)

The molecule workflow uses adaptive Metropolis-Hastings sampling with an
all-electron Gaussian proposal. Pretraining and training share these settings.

```{eval-rst}
.. config-defaults::
   :prefix: sampler
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

---

## Reference (`reference`)

Optional path to a `reference.npz` used for pretraining. When omitted, `train`
generates or reuses a PySCF reference. <project:reference.md> describes
resolution and standalone preparation.

In YAML:

```yaml
reference: ./runs/water/hf/reference.npz
```

Setting `pretrain.run.iterations=0` disables pretraining and reference
resolution, even when `reference` is set.

---

(train-stage)=
## Train Stage (`train.*`)

The main VMC optimization loop. Samples electron configurations, computes energy, and
updates wavefunction parameters.

(train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.run
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

(train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. Effective defaults for the built-in
  optimizers are listed below.

#### KFAC options

```{eval-rst}
.. config-defaults::
   :prefix: train.optim
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
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

(train-writers)=
### Writers (`train.writers.*`)

The train stage enables `console`, `csv`, and `npy` writers by default.
`csv` and `npy` write `train_stats.csv` and `train_stats.npy` in
`workflow.save_path` and take no extra configuration keys; set
`train.writers.csv=null` or `train.writers.npy=null` to disable them.

#### Console writer (`train.writers.console.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.writers.console
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

(train-grads)=
### Loss gradients (`train.grads.*`)

Loss and gradient estimator. Computes the VMC loss and parameter gradients.

```{eval-rst}
.. config-defaults::
   :prefix: train.grads
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

---

## Pretrain Stage (`pretrain.*`)

Initializes the neural network to approximate the loaded reference orbitals
before VMC training. Its run and writer settings follow the same schemas as the
train stage, while the optimizer default and workflow-wired supervised loss are
specific to pretraining.

### Sampling mixture (`pretrain.sample_fraction`)

Mixing fraction between the neural ansatz and the loaded reference during
pretrain sampling. `0.0` is pure neural, `1.0` (the default) is pure reference.

### Run options (`pretrain.run.*`)

```{eval-rst}
.. config-defaults::
   :prefix: pretrain.run
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

### Optimizer (`pretrain.optim.*`)

- Default optimizer module: `optax:adam`, and its effective keys are listed
  below.

#### Effective Adam defaults

```{eval-rst}
.. config-defaults::
   :prefix: pretrain.optim
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
   :scope: Adam
```

### Writers (`pretrain.writers.*`)

The pretrain stage enables `console`, `csv`, and `npy` writers by default.
`csv` and `npy` write `pretrain_stats.csv` and `pretrain_stats.npy` in
`workflow.save_path` and take no extra configuration keys; set
`pretrain.writers.csv=null` or `pretrain.writers.npy=null` to disable them.

#### Console writer (`pretrain.writers.console.*`)

```{eval-rst}
.. config-defaults::
   :prefix: pretrain.writers.console
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```

---

(train-estimators)=
## Estimators (`estimators.*`)

Training and evaluation share the same estimator collection, with training exposing `loss_weight` settings for VMC gradient and penalty computation (see [Loss and gradient](../../guide/estimators/loss-grad.md) and [How Estimators Work](../../guide/estimators/index.md)) and keeping the total-energy estimator always enabled. See the [evaluation estimator config](#molecule-estimators) for evaluation defaults.

```{eval-rst}
.. config-defaults::
   :prefix: estimators
   :defaults-from: jaqmc.app.molecule.workflow.MoleculeTrainWorkflow
```
