# Writers

Writers record training statistics at each step. JaQMC includes console, CSV,
HDF5, and Weights & Biases writers. Install the `wandb` package separately in
the same environment as JaQMC before using the W&B writer. See
<project:../guide/writers.md> for background on output files and configuring
writers.

## Configuration

For writer config keys, see the configuration reference: [Molecule](#train-writers), [Solid](#solid-train-writers), or [Hall](#hall-train-writers).

## Base class

```{eval-rst}
.. autoclass:: jaqmc.writer.base.Writer
   :members:

.. autoclass:: jaqmc.writer.base.Writers
   :members:
```

## Built-in writers

```{eval-rst}
.. autoclass:: jaqmc.writer.console.ConsoleWriter
   :members:
   :show-inheritance:

.. autoclass:: jaqmc.writer.console.FieldSpec
   :members: parse

.. autoclass:: jaqmc.writer.csv.CSVWriter
   :members:
   :show-inheritance:

.. autoclass:: jaqmc.writer.hdf5.HDF5Writer
   :members:
   :show-inheritance:

.. autoclass:: jaqmc.writer.wandb.WandbWriter
   :members:
   :show-inheritance:
```
