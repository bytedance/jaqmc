# Guides

Use this section when you already picked a system and want to run, analyze, tune, or debug a workflow. If you are new to JaQMC, start with <project:../getting-started/quick-start.md>.

## Run Workflows

- <project:running-workflows.md> — debug runs, production runs, resume/branch/evaluate, and reporting checklist.
- <project:configuration.md> — CLI overrides, YAML layering, and `--dry-run`.

## Analyze Results

- <project:training-stats.md> — read and visualize `train_stats.*` for optimization diagnostics.
- <project:analyzing-evaluations.md> — extract final observables and uncertainty from evaluation outputs.
- <project:analyzing-wavefunctions.md> — inspect learned wavefunction behavior and diagnostics.

## Tune Components

- <project:wavefunction.md> — choose and tune built-in wavefunction architectures.
- <project:optimizers/index.md> — pick and configure optimization methods.
- <project:sampling.md> — tune MCMC behavior and acceptance.
- <project:writers.md> — control which statistics are written and where.

## Physics and Estimators

- <project:estimators/index.md> — estimator physics, formulas, and computational details.
- <project:periodic-boundaries.md> — PBC concepts and practical implications for solids.

## Scale and Reliability

- <project:multi-device.md> — run across multiple GPUs/devices.
- <project:troubleshooting.md> — common failures, NaNs, and recovery paths.

```{toctree}
:hidden:
:maxdepth: 2

configuration.md
running-workflows.md
wavefunction.md
estimators/index.md
optimizers/index.md
sampling.md
periodic-boundaries.md
writers.md
multi-device.md
training-stats.md
analyzing-evaluations.md
analyzing-wavefunctions.md
troubleshooting.md
```
