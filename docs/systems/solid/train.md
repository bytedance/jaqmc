# Solid Training

Configuration reference for `jaqmc solid train`.
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
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
```

(solid-train-system)=
## System (`system.*`)

Defines the periodic solid system to simulate. The implementation is selected by
`system.module`.

- Default module selection: `base:SolidConfig`, so `system.*` is read
  directly as an arbitrary crystal config when `system.module` is omitted.
  Built-in choices are:
  - `base:SolidConfig` (or omitted): arbitrary crystal config
  - `rock_salt`: rock-salt crystal generator
  - `two_atom_chain`: two-atom chain generator

### Arbitrary crystals (default)

```{eval-rst}
.. config-defaults::
   :prefix: system
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
   :scope: Solid
```

### Rock salt (`system.module=rock_salt`)

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.rock_salt.RockSaltBuilder
   :prefix: system
   :scope: rock-salt
```

### Two-atom chain (`system.module=two_atom_chain`)

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.two_atom_chain.TwoAtomChainBuilder
   :prefix: system
   :scope: two-atom-chain
```

(solid-train-wf)=
## Wavefunction (`wf.*`)

- Default module selection: `SolidWavefunction` (`jaqmc.app.solid.wavefunction:SolidWavefunction`). Its effective defaults are listed below.

```{eval-rst}
.. config-defaults::
   :prefix: wf
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
   :scope: Solid
```

(solid-train-sampler)=
## Sampler (`sampler.*`)

The solid workflow uses adaptive Metropolis-Hastings sampling with a periodic
Gaussian proposal. Pretraining and training share these settings.

```{eval-rst}
.. config-defaults::
   :prefix: sampler
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
```

---

## Reference (`reference`)

Optional path to a `reference.npz`. When omitted, `train` reuses one from the
run directory or generates one with PySCF; the solid ansatz loads its occupied
k-points from the file even when `pretrain.run.iterations=0`.
<project:reference.md> describes resolution and standalone preparation.

In YAML:

```yaml
reference: ./runs/lih_solid/hf/reference.npz
```

Or as a CLI override: `reference=./runs/lih_solid/hf/reference.npz`.

---

(solid-train-stage)=
## Train Stage (`train.*`)

The main VMC optimization loop. Samples electron configurations, computes energy, and updates wavefunction parameters.

(solid-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.run
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
```

(solid-train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. Effective defaults for the built-in
  optimizers are listed below.

#### KFAC options

```{eval-rst}
.. config-defaults::
   :prefix: train.optim
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
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

(solid-train-writers)=
### Writers (`train.writers.*`)

The train stage enables `console`, `csv`, and `npy` writers by default.
`csv` and `npy` write `train_stats.csv` and `train_stats.npy` in
`workflow.save_path` and take no extra configuration keys; set
`train.writers.csv=null` or `train.writers.npy=null` to disable them.

#### Console writer (`train.writers.console.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.writers.console
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
```

(solid-train-grads)=
### Loss gradients (`train.grads.*`)

Loss and gradient estimator. Computes the VMC loss and parameter gradients. See [Loss and gradient](../../guide/estimators/loss-grad.md) for the mathematical derivation and outlier clipping details.

```{eval-rst}
.. config-defaults::
   :prefix: train.grads
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
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
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
```

### Optimizer (`pretrain.optim.*`)

- Default optimizer module: `optax:adam`, and its effective keys are listed
  below.

#### Effective Adam defaults

```{eval-rst}
.. config-defaults::
   :prefix: pretrain.optim
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
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
   :defaults-from: jaqmc.app.solid.workflow.SolidTrainWorkflow
```

### Loss gradients

Pretraining does not use configurable `pretrain.grads.*` settings. The workflow
wires a supervised orbital-matching loss against the loaded reference.

---

## Estimators (`estimators.*`)

Training and evaluation share the same estimator schema and defaults; see the
[evaluation estimator config](#solid-estimators). During training, the
total-energy estimator is always enabled and cannot be toggled off.
