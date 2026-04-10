# Custom Writers

Write a custom writer when you need to record training statistics to a destination beyond console, CSV, or HDF5 — for example, a database, a monitoring dashboard, or a custom binary format.

## The Write Lifecycle

Writers follow a simple lifecycle managed by the training loop:

1. **`open()`** is called once when the stage starts. Set up resources here — open files, establish connections, create tables. In distributed runs, `open()` runs only on the master process, so you don't need to guard against multiple writers.

2. **`write(step, stats)`** is called every training step. `stats` is a flat dictionary containing the output of all estimators' `reduce()` — keys like `total_energy`, `pmove`, `energy:kinetic_var`, etc. Values are JAX/NumPy scalars; use `self.to_scalar(val)` to convert to Python floats if your destination requires it.

3. **`open()` cleanup** runs when the stage ends (after `yield`). Close file handles, flush buffers, disconnect.

4. **Resumption**: When training resumes from a checkpoint, `open()` receives `initial_step` — the step where training will restart. If your writer persists to a file, truncate any data at or beyond this point so stale entries from a previous (interrupted) run are discarded.

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

**`open`** manages the resource lifecycle. All I/O setup goes here — never in `__init__`. In distributed runs, multiple processes instantiate the writer during configuration, but only the master process enters `open()`. If you put file creation in `__init__`, every process would create (and fight over) the same files.

```python
    @contextmanager
    def open(self, working_dir, stage_name, initial_step=0):
        path = working_dir / f"{stage_name}_my_log.txt"
        self._file = open(path, "a")
        # If resuming, truncate stale entries
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

## Getting Started

- {class}`~jaqmc.writer.console.ConsoleWriter` — simplest writer. Shows `to_scalar()` usage and selective field display.
- {class}`~jaqmc.writer.csv.CSVWriter` — file-based writer. Shows `open()` with file handle management and header writing.
- {class}`~jaqmc.writer.hdf5.HDF5Writer` — chunked array writes. Shows `initial_step` handling for checkpoint truncation.

## See Also

- <project:/guide/writers.md> — background on output files and console configuration
- <project:/api-reference/writers.md> — base class and built-in writer API
