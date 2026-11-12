# Solid Evaluation

Configuration reference for `jaqmc solid evaluate`.
This page shows the evaluation workflow defaults. `--dry-run` prints the
resolved config for a specific run. Evaluation keys
for `run.*`, `sampler.*`, and `writers.*` live at the config root. Defaults are
resolved in this order: schema defaults, YAML config, then CLI
overrides. For training config, see <project:train.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control evaluation-wide settings and checkpoint loading.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

Must match the training run. The effective defaults and built-in module choices
are identical to the [training system config](#solid-train-system).

## Wavefunction (`wf.*`)

Architecture settings must match the training run, though `wf.precision` can
be changed (for example, to `standard`, see
<project:../../guide/mixed-precision.md>). The effective defaults and built-in
module choices are identical to the
[training wavefunction config](#solid-train-wf).

## Reference (`reference`)

Optional path to a `reference.npz`. When omitted, evaluation checks
`workflow.restore_path`, `workflow.source_path`, and `workflow.save_path`, in
that order. If none contains the file, evaluation generates one with PySCF in
`workflow.save_path`. An explicit `reference=` path used for training is also
required for evaluation; evaluation does not search the output directory
created by `jaqmc solid reference prepare`. <project:reference.md> describes
the complete resolution order.

```yaml
reference: ./runs/lih_solid/hf/reference.npz
```

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

Solid evaluation uses adaptive Metropolis-Hastings sampling with a periodic
Gaussian proposal.

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

(solid-estimators)=
## Estimators (`estimators.*`)

Training and evaluation share this estimator schema. Runtime dependencies such
as the wavefunction and supercell lattice are wired at startup and omitted from
YAML. For physics and derivations, see
<project:../../guide/estimators/index.md>. For the API, see
[Estimators](../../api-reference/estimators.md).

```{eval-rst}
.. config-defaults::
   :prefix: estimators
   :defaults-from: jaqmc.app.solid.workflow.SolidEvalWorkflow
```
