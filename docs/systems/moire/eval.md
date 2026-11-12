# Evaluation

Configuration reference for `jaqmc moire evaluate`.
This page shows the evaluation workflow defaults. `--dry-run` prints the
resolved config for a specific run. Evaluation has
only one stage, so stage keys (`run.*`, `sampler.*`, `writers.*`) live at the
config root rather than under a `train.*` prefix. Defaults are resolved in this
order: schema defaults, YAML config, then CLI overrides. For
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

The evaluation NPY writer is always enabled because its per-step statistics
are required for digest computation, and it writes to `evaluation_stats.npy`.
Other writers are disabled by default; add one by setting its `module` key, such
as `writers.console.module=console` or `writers.csv.module=csv`. The CSV writer
writes `evaluation_stats.csv` in `workflow.save_path` and takes no extra
configuration keys.

### Console writer (`writers.console.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.console.ConsoleWriter
   :prefix: writers.console
```

(moire-estimators)=
## Estimators (`estimators.*`)

Training and evaluation share this estimator schema. Runtime dependencies such
as the wavefunction, supercell lattice, and Hamiltonian prefactors are wired at
startup and omitted from YAML. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

```{eval-rst}
.. config-defaults::
   :prefix: estimators
   :defaults-from: jaqmc.app.moire.workflow.MoireEvalWorkflow
```
