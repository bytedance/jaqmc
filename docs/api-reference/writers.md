# Writers

Writers record training statistics at each step. JaQMC ships with console, CSV, and HDF5 writers. See <project:../guide/writers.md> for background on output files and customizing console output.

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
```
