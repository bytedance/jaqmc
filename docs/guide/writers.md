# Writers

Writers record statistics produced during training. The built-in writers are:

- **Console** — Prints selected fields to the terminal.
- **CSV** — Appends scalar statistics to a CSV file (e.g., `train_stats.csv`) in the output directory.
- **HDF5** — Appends all statistics (including array-valued fields) to an HDF5 file (e.g., `train_stats.h5`) in the output directory.

Which writers are active depends on the workflow. Use `--dry-run` to see the resolved config — the `writers` section shows what is enabled. To disable a writer, set it to `null`; to re-enable one that isn't active, set its module path:

```bash
train.writers.hdf5=null                              # disable HDF5 for train
train.writers.csv.module=jaqmc.writer.csv:CSVWriter   # enable CSV for train
pretrain.writers.console.interval=10                  # configure pretrain's console writer
```

## Console Output

The console writer prints a configurable set of fields every `interval` steps (default: every step). The field spec format is `[alias=]key[:format]`, separated by commas:

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

For most estimators, the key is the output key in the estimator code. If a custom estimator returns `{"observable_a": value}` from `evaluate_local` and uses the default reducer, the scalar mean is written as `observable_a` and the variance is written as `observable_a_var`.

If you're unsure which key to use, inspect an existing run's output files. The CSV header lists scalar keys that can be printed in the console:

```bash
head -n 1 runs/my-run/train_stats.csv
```

Array-valued outputs, such as histograms, do not appear in CSV and cannot be printed in the console. Use the HDF5 file to inspect those keys instead.

## Output Files

CSV and HDF5 writers produce files in the `workflow.save_path` directory. CSV captures scalar statistics; HDF5 captures all statistics including array-valued fields (e.g., histograms). When resuming from a checkpoint, both formats automatically truncate stale data from interrupted runs.

By default, their output paths come from the templates `{stage}_stats.csv` and `{stage}_stats.h5`. You can customize these with `path_template`; the only supported template field is `{stage}`:

```bash
train.writers.csv.path_template=logs/{stage}_metrics.csv
evaluation.writers.hdf5.path_template=artifacts/{stage}/stats.h5
```

See <project:training-stats.md> for how to work with these files.

## See Also

- **Configuration:** [Molecule](#train-writers), [Solid](#solid-train-writers), [Hall](#hall-train-writers)
- **Extending:** <project:/extending/custom-components/writers.md>
- **API reference:** <project:/api-reference/writers.md>
