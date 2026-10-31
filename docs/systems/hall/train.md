# Quantum Hall Training

Configuration reference for `jaqmc hall train`.
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
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
```

(hall-train-system)=
## System (`system.*`)

Defines the quantum Hall system on the Haldane sphere.

See <project:index.md> for physics background and usage examples.

```{eval-rst}
.. config-defaults::
   :prefix: system
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
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
.. config-defaults::
   :prefix: wf
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
   :scope: MHPO
```

(hall-train-sampler)=
## Sampler (`sampler.*`)

The Hall workflow uses adaptive Metropolis-Hastings sampling with a spherical
proposal.

```{eval-rst}
.. config-defaults::
   :prefix: sampler
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
```

(hall-train-stage)=
## Train Stage (`train.*`)

The VMC optimization loop. Samples electron configurations on the Haldane sphere, computes energy (and optional angular momentum penalties), and updates wavefunction parameters.

(hall-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.run
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
```

(hall-train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. Effective defaults for the built-in
  optimizers are listed below.

#### KFAC options

```{eval-rst}
.. config-defaults::
   :prefix: train.optim
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
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

The train stage enables `console`, `csv`, and `npy` writers by default.
`csv` and `npy` write `train_stats.csv` and `train_stats.npy` in
`workflow.save_path` and take no extra configuration keys; set
`train.writers.csv=null` or `train.writers.npy=null` to disable them.

#### Console writer (`train.writers.console.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.writers.console
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
```

### Loss gradients (`train.grads.*`)

The workflow resolves {class}`~jaqmc.estimator.loss_grad.LossGrad` from `train.grads.*`. See [Loss and gradient](../../guide/estimators/loss-grad.md) for the clipping formulas.

```{eval-rst}
.. config-defaults::
   :prefix: train.grads
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
```

---

(hall-train-estimators)=
## Estimators (`estimators.*`)

Training and evaluation share the same estimator collection, with training exposing `loss_weight` settings for VMC gradient and penalty computation (see [Angular momentum penalties](index.md#angular-momentum-penalties), [Loss and gradient](../../guide/estimators/loss-grad.md), and [How Estimators Work](../../guide/estimators/index.md)) and keeping the total-energy estimator always enabled. See the [evaluation estimator config](#hall-estimators) for evaluation defaults.

```{eval-rst}
.. config-defaults::
   :prefix: estimators
   :defaults-from: jaqmc.app.hall.workflow.HallTrainWorkflow
```
