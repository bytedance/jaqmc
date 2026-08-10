# Evaluation

Configuration reference for `jaqmc moire evaluate`.
This page shows the effective defaults for the evaluation workflow. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Evaluation has
only one stage, so stage keys (`run.*`, `sampler.*`, `writers.*`) live at the
config root rather than under a `train.*` prefix. Defaults are resolved in this
order: schema defaults, workflow preset, YAML config, then CLI overrides. For
training config, see <project:train.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control evaluation-wide settings and checkpoint loading.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

Must match the training run. The effective defaults are identical to the
[training system config](#moire-train-system).

## Wavefunction (`wf.*`)

Must match the training run. The effective defaults are identical to the
[training wavefunction config](#moire-train-wf). In particular, `wf.k_com` must
match the FCI sector used during training.

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

- Default sampler module: the moire joint Metropolis-Hastings sampler, matching
  training. Its effective keys are listed below.

```{eval-rst}
.. config-defaults:: jaqmc.app.moire.workflow.MoireJointMCMCSampler
   :prefix: sampler
```

## Writers (`writers.*`)

No external writers are enabled by default. If you enable them manually, the
root-level writer keys below control their configuration. The evaluation stage
always writes per-step statistics to an internal HDF5 file for digest
computation; this is independent of the writers configured here.

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

### HDF5 writer (`writers.hdf5.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.hdf5.HDF5Writer
   :prefix: writers.hdf5
```

## Estimators (`estimators.*`)

Energy estimator definitions match training. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

`estimators.enabled.energy` defaults to `true` and controls the whole energy
estimator set during evaluation. When energy is enabled, the following component
flags control which terms are included; each also defaults to `true`:

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
