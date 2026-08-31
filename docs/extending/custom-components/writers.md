# Custom Writers

Write a custom writer when you need to record training statistics to a destination beyond console, CSV, or HDF5 — for example, a database, a monitoring dashboard, or a custom binary format.

## The Write Lifecycle

The stage drives writers in this order on the master process:

1. **`sync_history(source_dir, working_dir, stage_name, steps)`** runs before the writer opens. `source_dir` is the restore directory (`workflow.restore_path`, or its parent if that path is a file) and `working_dir` is `workflow.save_path`. `steps` is the step the stage will resume from (`0` on a fresh run). If you persist a step log — a file, database, or other store — copy history from `source_dir` into `working_dir` when those directories differ, then keep only records for steps `0 .. steps-1`. Console and other writers that do not persist history can leave the default no-op.

2. **`open(working_dir, stage_name)`** is called once when the stage starts. Set up resources here — open files, establish connections, create tables. In distributed runs, `open()` runs only on the master process, so you don't need to guard against multiple writers.

3. **`write(step, stats)`** is called every training step. `stats` is a flat dictionary of that step's recorded values — estimator outputs and, during training, sampler diagnostics such as `pmove`. Values may be JAX/NumPy scalars or arrays; use `self.to_scalar(val)` when your destination needs Python floats.

4. **`open()` cleanup** runs when the stage ends (after `yield`). Close file handles, flush buffers, disconnect.

The user-facing resume and branch recipes that produce a different `working_dir` are in <project:/guide/running-workflows.md#recipe-resume-evaluate>.

## Building a Writer

Subclass {class}`~jaqmc.writer.base.Writer`:

```python
from contextlib import contextmanager

from jaqmc.writer.base import Writer
from jaqmc.utils.config import configurable_dataclass

@configurable_dataclass
class MyWriter(Writer):
    log_dir: str = "/tmp/logs"  # config field — tunable via YAML
```

**`sync_history`** prepares persisted history before `open()`. Skip this method only if the writer does not persist a step log.

```python
    def sync_history(self, source_dir, working_dir, stage_name, steps):
        filename = f"{stage_name}_my_log.txt"
        src = source_dir / filename
        dest = working_dir / filename
        if source_dir != working_dir and src.exists():
            dest.write_text(src.read_text())
        if not dest.exists():
            return
        lines = dest.read_text().splitlines()[:steps]
        dest.write_text(("\n".join(lines) + "\n") if lines else "")
```

**`open`** manages the resource lifecycle. All I/O setup goes here — never in `__init__`. In distributed runs, multiple processes instantiate the writer during configuration, but only the master process enters `open()`. If you put file creation in `__init__`, every process would create (and fight over) the same files.

```python
    @contextmanager
    def open(self, working_dir, stage_name):
        path = working_dir / f"{stage_name}_my_log.txt"
        self._file = open(path, "a")
        try:
            yield
        finally:
            self._file.close()
```

**`write`** records one step's statistics. Keep it fast — it runs every iteration inside the training loop:

```python
    def write(self, step, stats):
        energy = self.to_scalar(stats.get("total_energy", float("nan")))
        pmove = self.to_scalar(stats.get("pmove", float("nan")))
        self._file.write(f"{step},{energy},{pmove}\n")
```

{class}`~jaqmc.writer.csv.CSVWriter` and {class}`~jaqmc.writer.hdf5.HDF5Writer` apply the same copy-then-truncate pattern to `{stage}_stats.*`.

## Getting Started

- {class}`~jaqmc.writer.console.ConsoleWriter` — simplest writer. Shows `to_scalar()` usage and selective field display.
- {class}`~jaqmc.writer.csv.CSVWriter` — file-based writer. Shows `open()` with file handle management, header writing, and `sync_history` for resume.
- {class}`~jaqmc.writer.hdf5.HDF5Writer` — chunked array writes. Shows `sync_history` for copy into a new directory and checkpoint truncation.
- {class}`~jaqmc.writer.wandb.WandbWriter` — external service writer. Shows scalar filtering, W&B run lifecycle, and an optional dependency that must be installed separately.

## See Also

- <project:/guide/writers.md> — background on output files and console configuration
- <project:/api-reference/writers.md> — base class and built-in writer API
