# Writers

Writers record per-step statistics. The built-in writers are:

- **Console** — Prints selected fields to the terminal.
- **CSV** — Appends scalar statistics to a CSV file (e.g., `train_stats.csv`) in the output directory.
- **HDF5** — Appends all statistics (including array-valued fields) to an HDF5 file (e.g., `train_stats.h5`) in the output directory.
- **W&B** — Sends scalar statistics to Weights & Biases.

Which configurable writers are active depends on the workflow. Use `--dry-run`
to see them in the resolved `writers` section. To disable a configurable writer,
set it to `null`; to re-enable one that isn't active, set its module path.

Evaluation writes `evaluation_stats.h5` as a required analysis file, not through
the configurable HDF5 writer. Console and CSV stay off during evaluation unless
you enable them at the config root (`writers.console.*`, `writers.csv.*`), not
under `train.writers.*`.

```bash
train.writers.hdf5=null                               # disable HDF5 for train
train.writers.csv.module=jaqmc.writer.csv             # enable CSV for train
train.writers.wandb.module=jaqmc.writer.wandb         # enable W&B for train
pretrain.writers.console.interval=10                  # configure pretrain's console writer
writers.console.module=jaqmc.writer.console           # enable console during evaluate
writers.csv.module=jaqmc.writer.csv                   # enable CSV during evaluate
```

## Console Output

The console writer prints a configurable set of fields every `interval` steps
(default: every step). The field spec format is `[alias=]key[:format]`,
separated by commas. The examples below use the training prefix; for
`jaqmc <app> evaluate`, use `writers.console.*` instead of
`train.writers.console.*`.

```bash
# Customize precision for energy and variance
train.writers.console.fields="pmove:.2f,energy=total_energy:.6f,variance=total_energy_var:.6f"

# Use an alias for a long stat key
train.writers.console.fields="E=total_energy:.6f,Lz=angular_momentum_z:+.4f"

# Print every 10 steps instead of every step
train.writers.console.interval=10
```

Use `--dry-run` to see the default fields for your workflow.

To add another console field, use the statistic key that the estimator writes, not the estimator's name. For example, if an estimator returns an `observable_a` statistic, add that key to the console fields:

```bash
train.writers.console.fields="pmove:.2f,energy=total_energy:.6f,A=observable_a:.6f"
```

For most estimators, the key is the output key in the estimator code. If a custom estimator returns `{"observable_a": value}` from `evaluate_single_walker` and uses the default reducer, the scalar mean is written as `observable_a` and the variance is written as `observable_a_var`.

If you're unsure which key to use, inspect an existing run's output files. The CSV header lists scalar keys that can be printed in the console:

```bash
head -n 1 runs/my-run/train_stats.csv
```

Array-valued outputs, such as histograms, do not appear in CSV and cannot be printed in the console. Use the HDF5 file to inspect those keys instead.

## Weights & Biases

The W&B writer logs real-valued scalar statistics to Weights & Biases. It is
not enabled by default.

Install `wandb` manually in the same Python environment where JaQMC is
installed:

```bash
uv pip install wandb
# or, after activating the JaQMC environment:
pip install wandb
```

Then enable the writer and set the W&B project metadata you want:

```bash
train.writers.wandb.module=jaqmc.writer.wandb:WandbWriter
train.writers.wandb.project=my-project
train.writers.wandb.run_name=my-run
```

When enabled, the W&B writer creates a W&B run for the stage and logs the stage
name as the W&B job type. In advanced programmatic setups, if a W&B run is
already active before JaQMC starts the stage, the writer reuses that run.

## Output Files

CSV and HDF5 writers produce files in the `workflow.save_path` directory. CSV captures scalar statistics; HDF5 captures all statistics including array-valued fields. Their output files are named `{stage}_stats.csv` and `{stage}_stats.h5`.

For resume and branch behavior, see
<project:running-workflows.md#recipe-resume-evaluate>. For reading the files,
see <project:training-stats.md>.

## See Also

- **Configuration:** [Molecule](#train-writers), [Solid](#solid-train-writers), [Hall](#hall-train-writers)
- **Extending:** <project:/extending/custom-components/writers.md>
- **API reference:** <project:/api-reference/writers.md>
