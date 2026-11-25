# Training

Configuration reference for `jaqmc moire train`.
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
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

(moire-train-system)=
## System (`system.*`)

Defines the moire supercell geometry and the continuum-model physical
parameters. User-facing values are in explicit physical units (lengths in nm,
energies in meV, phases in degrees); the config derives the dimensionless
supercell geometry and energy prefactors used by the sampler and estimators.

The default configuration is a fractional two-thirds-filling system. Select a
different system by overriding `system.*` fields (via `--yml` or inline, e.g.
`system.supercell_matrix="[[3,0],[0,3]]"`). See <project:index.md> for physics
background and the meaning of each parameter.

```{eval-rst}
.. config-defaults::
   :prefix: system
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

(moire-train-wf)=
## Wavefunction (`wf.*`)

Configures the layer-pseudospin neural-network ansatz. The phase mode
(`singlephase` for integer CI, `multiphase` for fractional FCI) is inferred
automatically from the filling and is not set directly. The main user-facing
knob is `wf.k_com`, the FCI center-of-mass momentum in simulation-cell
reciprocal fractional coordinates:

- Integer-filling CI: leave `wf.k_com` unset/`null` (setting `wf.k_com` in `singlephase` mode raises an error).
- Fractional-filling FCI: `wf.k_com` defaults to `[0.0, 0.0]` (the $\Gamma$
  sector) when unset, so translation projection is on by default. Select a
  different sector from YAML (a `k_com: [1.0, 1.0]` entry under the top-level
  `wf:` mapping) or a CLI override (`wf.k_com=[1.0,1.0]`).

See <project:index.md> for background on continuous spin, phase modes, and
translation projection.

```{eval-rst}
.. config-defaults::
   :prefix: wf
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

(moire-train-sampler)=
## Sampler (`sampler.*`)

Training and evaluation use the same root-level sampler schema.

- Default sampler module: the moire joint Metropolis-Hastings sampler, which
  proposes spatial and spin-angle moves together. Its moire-tuned defaults
  (`steps=20`, `initial_width=0.02`) mix better in moire units than the generic
  MCMC defaults.

```{eval-rst}
.. config-defaults::
   :prefix: sampler
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

(moire-train-stage)=
## Train Stage (`train.*`)

The VMC optimization loop. Samples electron positions and continuous spin
angles with a joint proposal, computes the moire local energy, and updates the
wavefunction parameters.

(moire-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.run
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

(moire-train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. Effective defaults for KFAC are listed
  below.

#### KFAC options

```{eval-rst}
.. config-defaults::
   :prefix: train.optim
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
   :scope: KFAC
```

#### SR options

Selecting another optimizer replaces the KFAC configuration with that module's
own dataclass defaults. Only fields accepted by the selected module may be
overridden (for example, Optax optimizers do not accept `train.optim.damping`).

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

(moire-train-writers)=
### Writers (`train.writers.*`)

The train stage enables `console`, `csv`, and `npy` writers by default.
`csv` and `npy` write `train_stats.csv` and `train_stats.npy` in
`workflow.save_path` and take no extra configuration keys; set
`train.writers.csv=null` or `train.writers.npy=null` to disable them.

#### Console writer (`train.writers.console.*`)

```{eval-rst}
.. config-defaults::
   :prefix: train.writers.console
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

### Loss gradients (`train.grads.*`)

The workflow resolves {class}`~jaqmc.estimator.loss_grad.LossGrad` from `train.grads.*`. Its default gradient clipping window has scale 20. See [Loss and gradient](../../guide/estimators/loss-grad.md) for the clipping formulas.

```{eval-rst}
.. config-defaults::
   :prefix: train.grads
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```

---

(moire-train-estimators)=
## Estimators (`estimators.*`)

Training and evaluation share the same estimator collection, with training exposing `loss_weight` settings for VMC gradient and penalty computation (see [Loss and gradient](../../guide/estimators/loss-grad.md) and [How Estimators Work](../../guide/estimators/index.md)) and keeping the total-energy estimator always enabled. See the [evaluation estimator config](#moire-estimators) for evaluation defaults.

```{eval-rst}
.. config-defaults::
   :prefix: estimators
   :defaults-from: jaqmc.app.moire.workflow.MoireTrainWorkflow
```
