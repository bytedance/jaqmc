# Quantum Hall Evaluation

Configuration reference for `jaqmc hall evaluate`.
This page shows the evaluation workflow defaults. `--dry-run` prints the
resolved config for a specific run. Evaluation keys
for `run.*`, `sampler.*`, and `writers.*` live at the config root. Defaults are
resolved in this order: schema defaults, YAML config, then CLI
overrides. For training config, see <project:train.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control evaluation-wide settings and checkpoint loading.

`workflow.source_path` selects the trained parameters and sampler state to
evaluate. It is required for MHPO and other trainable wavefunctions. The
analytic `laughlin` and `free` wavefunctions have no parameters, so they can
start evaluation from fresh walkers without a source path.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

When evaluating a training checkpoint, these settings must match the training
run. For direct `laughlin` or `free` evaluation, they define the target system.
The effective defaults are identical to the [training system config](#hall-train-system).

## Wavefunction (`wf.*`)

When evaluating a training checkpoint, architecture settings must match the
training run, though `wf.precision` can be changed (for example, to
`standard`, see <project:../../guide/mixed-precision.md>). For direct
`laughlin` or `free` evaluation, select the analytic module with `wf.module`.
MHPO options match the [training wavefunction config](#hall-train-wf).

### Laughlin options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.laughlin.Laughlin
   :prefix: wf
   :scope: Laughlin
```

### Free options (`wf.*`)

The `free` monopole-harmonics wavefunction takes no extra `wf.*` configuration
keys.

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

Hall evaluation uses adaptive Metropolis-Hastings sampling with a spherical
proposal.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
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

(hall-estimators)=
(hall-angular-projection)=
## Estimators (`estimators.*`)

Training and evaluation share this estimator collection, with evaluation omitting training `loss_weight` fields (see [training estimators](#hall-train-estimators)). Runtime dependencies such
as the wavefunction and sphere geometry are wired at startup and omitted from
YAML. For physics, derivations, and post-processing (including
[angular-momentum sector projection](../../guide/estimators/angular-momentum.md)
and pair-correlation normalization), see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

```{eval-rst}
.. config-defaults::
   :prefix: estimators
   :defaults-from: jaqmc.app.hall.workflow.HallEvalWorkflow
```
