# Running Workflows

This page explains the shared mechanics of running JaQMC workflows: how to do a quick
debug run, how to launch a real training job, how resume and evaluation work, and what
files each command writes.

This page is for users running built-in apps such as `jaqmc molecule`, `jaqmc solid`, and
`jaqmc hall`. If you are building a new app/workflow implementation, see
<project:../extending/writing-workflows.md>.

In command examples, replace `<app>` with your system command name such as `molecule`,
`solid`, or `hall`.

## Common Tasks

Most users arrive here with one of four goals. Start with the recipe that matches what
you want to do, then use the later sections as reference.

(recipe-fast-debug-run)=
### Fast debug run

Use this when you want to confirm that the workflow starts, writes outputs, and makes
progress without waiting for a full calculation.

1. Pick your normal system input: CLI flags or one or more `--yml` files.
2. Set an explicit save path so the short run does not mix with other outputs.
3. Cut the training budget down to something that finishes quickly.
4. If your app has a pretrain stage, shorten that stage too.
5. Lower `workflow.batch_size` for speed, for example to `256` or `512`.
6. If needed, reduce model size with your app's `wf.*` knobs.

```bash
jaqmc <app> train ... \
  workflow.save_path=./runs/<name>-debug \
  train.run.iterations=100
```

What success looks like:

- `workflow.save_path` is created and contains `config.yaml`.
- The logs show increasing step numbers.
- Reported losses and energies stay finite.
- A final checkpoint is written.

If the run fails to initialize or produces NaNs, continue in <project:troubleshooting.md>.

### Production baseline

Use this when you are launching a real optimization run on a workstation or cluster.

1. Start from the recommended settings on your system page and its config reference.
2. Set a stable, descriptive `workflow.save_path`.
3. Tune the optimization budget first: usually `pretrain.run.iterations` and
   `train.run.iterations`.
4. Keep `workflow.batch_size` at its default unless you have evidence that the training
   signal is too noisy or your hardware budget requires a change.
5. Leave lower-level knobs alone until you have a specific reason to touch them.

```bash
jaqmc <app> train ... workflow.save_path=./runs/<name>-prod
```

For most production runs, the highest-value decisions are how long to train, whether to
keep pretraining, and whether you need a different optimizer or wavefunction. Everything
else is usually a second-order adjustment.

#### Fit larger runs in memory

If a run does not fit in device memory, first decide whether you can change the walker
count. Lowering `workflow.batch_size` reduces memory, but it also increases variance and
can hurt optimization quality. To keep the same number of walkers, set `vmap_chunk_size`
on a local estimator so the work is evaluated in smaller pieces. That trades speed for
lower peak memory.

The tightest part of the model varies by system, but the kinetic energy estimator is
often the right place to start:

```bash
jaqmc <app> train ... \
  estimators.energy.kinetic.vmap_chunk_size=128
```

Use the largest chunk size that still fits. Smaller chunks cut peak memory further, but
they add overhead.

(recipe-resume-evaluate)=
### Resume, branch, or evaluate

Use this when you already have a run on disk and want to continue it, fork from it, or
measure final observables from its checkpoints.

Resume training in the same directory by reusing `workflow.save_path` and increasing the
training budget:

```bash
jaqmc <app> train ... \
  workflow.save_path=./runs/<name>-prod \
  train.run.iterations=<larger_value>
```

Branch from an existing run into a new output directory by setting both save and restore
paths:

```bash
jaqmc <app> train ... \
  workflow.save_path=./runs/<name>-v2 \
  workflow.restore_path=./runs/<name>-prod \
  train.run.iterations=<larger_value>
```

Run evaluation in a separate directory by pointing `workflow.source_path` at the training
run you want to evaluate:

```bash
jaqmc <app> evaluate ... \
  workflow.save_path=./runs/<name>-eval \
  workflow.source_path=./runs/<name>-prod \
  run.iterations=1000
```

If you want multiple evaluation variants, give each one its own `workflow.save_path` and
reuse the same `workflow.source_path`.

```{tip}
These three path settings do different jobs:

- `workflow.save_path`: where the current command writes outputs
- `workflow.restore_path`: where the current command looks for its own checkpoints; if
  unset, it defaults to `workflow.save_path`
- `workflow.source_path`: training run or checkpoint used by `evaluate` to load trained
  parameters, walker data, and sampler state
```

When you rerun a command with the same `workflow.save_path`, JaQMC resumes from the
latest checkpoint already in that directory. Use `workflow.restore_path` only when you
want to start writing into a different output directory while restoring state from
somewhere else.

### Reporting checklist

Use this checklist before sharing final numbers:

1. Prefer evaluation outputs over in-training values for final reported observables.
2. Estimate uncertainty from per-step evaluation data, not from a single digest value.
3. Keep the resolved config and run provenance with the result.
4. Record any non-default estimator, optimizer, or wavefunction settings.
5. Make sure comparisons across runs use the same observable definitions and units.

For detailed analysis workflows, see <project:training-stats.md> and
<project:analyzing-evaluations.md>.

## Shared Command Model

The exact stage structure is app-defined, so this page does not try to describe one
universal training pipeline. System pages are the source of truth for what a given app
actually runs.

What *is* shared across apps is the command contract:

- **`jaqmc <app> train`** runs that app's training workflow and writes its outputs under
  `workflow.save_path`.
- **`jaqmc <app> evaluate`** runs that app's evaluation workflow, loading trained state
  from `workflow.source_path` and writing evaluation outputs under its own
  `workflow.save_path`.
- Training-stage keys are workflow-specific, so their structure lives in the app's config
  reference rather than on this page.
- Evaluation-stage keys live at the config root, which is why evaluation examples use
  `run.*`, `sampler.*`, and `writers.*` directly.

```{tip}
The config scope changes between these commands. In particular, evaluation stage keys
live at the config root, so use `run.iterations=...` with `jaqmc <app> evaluate`, not
`train.run.iterations=...`.
```

If you need the exact workflow shape for a specific app, go back to that system page. For
authoritative key definitions, use the config reference for your system:
[Molecule](../systems/molecule/train.md), [Solid](../systems/solid/train.md), or
[Hall](../systems/hall/train.md). For general override rules and YAML composition, see
<project:configuration.md>.

## Paths, Outputs, And Checkpoints

All outputs for a command go into `workflow.save_path`. If you do not set it explicitly,
JaQMC creates a timestamped directory under `runs/`. If the current working directory is inside the source repo, that means
the repo-level `runs/` directory; outside the repo, it means `./runs/` in your current
working directory.

Typical training output looks like this. If your workflow does not include a pretrain
stage, the `pretrain_*` files are simply absent:

```
save_path/
├── config.yaml
├── pretrain_ckpt_NNNNNN.npz
├── pretrain_stats.h5
├── pretrain_stats.csv
├── train_ckpt_NNNNNN.npz
├── train_stats.h5
└── train_stats.csv
```

Typical evaluation output looks like this:

```
save_path/
├── config.yaml
├── evaluation_ckpt_NNNNNN.npz
├── evaluation_stats.h5
├── evaluation_stats.csv        # optional, depends on configured writers
└── evaluation_digest.npz
```

A few details are worth keeping straight:

- `*_ckpt_*.npz` files are resumable checkpoints for that stage.
- `*_stats.h5` stores per-step statistics in a machine-friendly format.
- `*_stats.csv` is writer-dependent, so it may or may not be present.
- `evaluation_digest.npz` is the compact summary produced at the end of evaluation.

Evaluation loads from `workflow.source_path`. If that path is a directory, JaQMC restores
the latest `train_ckpt_*.npz` it finds there. If it is a specific checkpoint file, JaQMC
uses that file directly.

Checkpoint saves follow one rule: the last iteration is always saved, and intermediate
checkpoints are written only when both `save_time_interval` and `save_step_interval` are
satisfied. With the defaults (`600` seconds and every `1000` steps), that means a fast
job may skip step-1000 checkpointing if 10 minutes have not passed yet, but it still
writes a final checkpoint when the run ends.

For final observables, use evaluation output rather than training logs. In practice,
`evaluation_digest.npz` is the quick summary, while `evaluation_stats.h5` is the file you
need for uncertainty analysis from per-step data. For a full walkthrough, see
<project:analyzing-evaluations.md>.

If you are implementing a new workflow or app rather than running an existing one, continue
with <project:../extending/writing-workflows.md> for the framework-side wiring model.
