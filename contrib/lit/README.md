# JaQMC contributed LIT

Optional electric-dipole Lorentz integral transform workflows for JaQMC.

This is an experimental contributed package maintained independently from the
JaQMC core API. Install it from the workspace with:

```bash
uv sync --package jaqmc-contrib-lit
source .venv/bin/activate
```

The distribution registers the `jaqmc lit` command group through JaQMC's
`jaqmc.apps` contribution interface. Use `jaqmc lit run` to create a raw
spectrum and `jaqmc lit invert` for CPU postprocessing. Standard JaQMC
workflows do not import this package when it is not installed.

The complete He z-axis input example is in `examples/he`. It includes the
production YAML files and scripts for the GPU calculation, inversion, tables,
report, and figures. Generated results are written under `runs/he_lit` and are
not included in the package source.

## Execution limits

LIT runs must use exactly one JAX process. Set `lit.parallel.mode=off` for a
single-device calculation or `lit.parallel.mode=local_devices` for one process
controlling multiple local devices. Multi-process and multi-host launches are
rejected before the workflow creates source pools, checkpoints, or spectrum
files.

Detailed usage and scientific assumptions are documented in `docs/index.md`.
