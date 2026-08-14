# Training

Configuration reference for `jaqmc moire train`.
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
   :preset: jaqmc.app.moire.workflow.MoireTrainWorkflow.default_preset
```

## Workflow (`workflow.*`)

These keys control workflow-level settings shared across all stages.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.base.WorkflowConfig
   :prefix: workflow
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
.. config-defaults:: jaqmc.app.moire.config.MoireConfig
   :prefix: system
```

(moire-train-wf)=
## Wavefunction (`wf.*`)

Configures the layer-pseudospin neural-network ansatz. The phase mode
(`singlephase` for integer CI, `multiphase` for fractional FCI) is inferred
automatically from the filling and is not set directly. The main user-facing
knob is `wf.k_com`, the FCI center-of-mass momentum in simulation-cell
reciprocal fractional coordinates:

- Integer-filling CI: leave `wf.k_com` unset (it is ignored in `singlephase`).
- Fractional-filling FCI: `wf.k_com` defaults to `[0.0, 0.0]` (the $\Gamma$
  sector) when unset, so translation projection is on by default. Select a
  different sector from YAML (a `k_com: [1.0, 1.0]` entry under the top-level
  `wf:` mapping) or a CLI override (`wf.k_com=[1.0,1.0]`).

See <project:index.md> for background on continuous spin, phase modes, and
translation projection.

```{eval-rst}
.. config-defaults:: jaqmc.app.moire.wavefunction.MoireWavefunction
   :prefix: wf
```

(moire-train-sampler)=
## Sampler (`sampler.*`)

Training and evaluation use the same root-level sampler schema.

- Default sampler module: the moire joint Metropolis-Hastings sampler, which
  proposes spatial and spin-angle moves together. Its moire-tuned defaults
  (`steps=20`, `initial_width=0.02`) mix better in moire units than the generic
  MCMC defaults.

```{eval-rst}
.. config-defaults:: jaqmc.app.moire.workflow.MoireJointMCMCSampler
   :prefix: sampler
```

(moire-train-stage)=
## Train Stage (`train.*`)

The VMC optimization loop. Samples electron positions and continuous spin
angles with a joint proposal, computes the moire local energy, and updates the
wavefunction parameters.

(moire-train-run)=
### Run options (`train.run.*`)

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
   :prefix: train.run
```

(moire-train-optim)=
### Optimizer (`train.optim.*`)

- Default optimizer module: `kfac`. The workflow preset supplies the moire-tuned
  learning rate ($3\times10^{-3}$, delay $10^4$) and damping
  ($3\times10^{-4}$) as `train.optim.*` values, so the KFAC table below lists
  them as the effective defaults.

#### KFAC options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.kfac.kfac.KFACOptimizer
   :prefix: train.optim
   :scope: KFAC
```

#### SR options

Because the moire tuning is a preset rather than an optimizer subclass,
`train.optim.learning_rate` and `train.optim.damping` are also applied to
whichever module `train.optim.module` selects:

- `jaqmc.optimizer.sr:SROptimizer` has both fields, so it starts from the
  moire-tuned values above instead of the SR defaults listed below.
- The optax optimizers (`jaqmc.optimizer.optax:adam`, `:sgd`, ...) have no
  `damping` field and abort with
  `SerdeError: Invalid config at 'train.optim': unknown fields: {'damping'}`.

Clear the preset section to get a module's own defaults: a `train.optim: null`
entry in a YAML file drops the preset keys, and CLI overrides then repopulate
the section.

```yaml
# clear_optim.yml
train:
  optim: null
```

```bash
jaqmc moire train --yml clear_optim.yml \
    train.optim.module=jaqmc.optimizer.optax:adam
```

The SR and Adam tables below show their own module defaults, that is, the values
that apply once the preset section is cleared.

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.sr.SROptimizer
   :prefix: train.optim
   :scope: SR
   :preset:
```

#### Adam options

```{eval-rst}
.. config-defaults:: jaqmc.optimizer.optax.adam
   :prefix: train.optim
   :scope: Adam
   :preset:
```

(moire-train-writers)=
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
from `train.grads.*`. The preset uses a gradient clipping window of 20. See
[Loss and gradient](../../guide/estimators/loss-grad.md) for the clipping
formulas.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.loss_grad.LossAndGrad
   :prefix: train.grads
```

---

(moire-estimators)=
## Estimators (`estimators.*`)

Energy estimators are assembled programmatically by the workflow. The same
definitions are used by <project:eval.md>. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

Training always enables the energy estimator set when it calls the estimator
factory, so `estimators.enabled.energy` does not control training. The following
component flags control which terms are included; each defaults to `true`:

- `estimators.enabled.kinetic`
- `estimators.enabled.coulomb`
- `estimators.enabled.moire_potential`
- `estimators.enabled.soc`

`TotalEnergy` is added by the workflow, automatically sums the enabled
`energy:`-prefixed components, and is not object-configurable. The workflow also
constructs `CoulombInteractionEnergy`, `MoirePotential`, and `MoireSOC`
directly; their object fields are not configured through corresponding
`estimators.energy.*` keys. The kinetic estimator remains configurable as shown
below.

### Kinetic energy (`estimators.energy.kinetic.*`)

Common Euclidean kinetic term $-\tfrac{1}{2m^*}\sum_i \nabla_i^2\Psi/\Psi$. The
valley-momentum shifts are added separately by the SOC estimator.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.EuclideanKinetic
   :prefix: estimators.energy.kinetic
```

### Coulomb interaction

`CoulombInteractionEnergy` evaluates the electron-electron interaction using 2D
Ewald summation with a uniform neutralizing background. The workflow derives it
from the simulation lattice and `system.dielectric_constant` /
`system.moire_lattice_constant_nm`; it is not object-configurable.

### Moire potential

`MoirePotential` evaluates the layer-pseudospin moire potential
$\Delta_b, \Delta_t, \Delta_T$. The workflow derives it from
`system.moire_lattice_vectors`, `system.electron_spins`, `system.v1_mev`,
`system.phi1_deg`, and `system.omega1_mev`; it is not object-configurable.

### Valley-momentum shift / SOC

`MoireSOC` promotes the common kinetic term to the shifted operators
$(-i\nabla - \eta\,\mathbf{K}_\pm)^2/2$. The workflow derives it from
`system.moire_lattice_vectors`, `system.electron_spins`, `system.effective_mass`,
and `system.moire_lattice_constant_nm`; it is not object-configurable.
