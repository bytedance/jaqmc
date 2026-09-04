# He z-axis LIT input example

This directory contains the inputs and executable scripts for an end-to-end He
electric-dipole LIT calculation. No checkpoints, raw spectra, logs, fitted
results, tables, reports, or figures are checked in.

## Contents

- `config/ground.yml`: He ground-state training configuration.
- `config/lit.yml`: eight-GPU z-axis LIT configuration.
- `config/invert.yml`: one-pole, constant-background inversion configuration.
- `scripts/run_calculation.sh`: ground-state and LIT GPU calculation.
- `scripts/reproduce_postprocessing.sh`: inversion, tables, report, and figures.
- `scripts/post.py`: deterministic report and plotting code.

All generated files are written under `runs/he_lit` by default. Set
`JAQMC_LIT_RUN_ROOT` to use a different output directory.

## Install

From the JaQMC repository root:

```bash
python -m pip install -e .
python -m pip install -e "contrib/lit[reproduction]"
```

The `jaqmc lit` command should then be visible in `jaqmc --help`.

## Validate the inputs without GPUs

```bash
jaqmc molecule train \
  --yml contrib/lit/examples/he/config/ground.yml \
  --dry-run

jaqmc lit run \
  --yml contrib/lit/examples/he/config/lit.yml \
  --dry-run

jaqmc lit invert \
  --yml contrib/lit/examples/he/config/invert.yml \
  --dry-run
```

## Run the calculation

The configuration uses eight visible GPUs, 1024 response walkers per device,
and a 601-point frequency grid. The full calculation is expensive.

```bash
contrib/lit/examples/he/scripts/run_calculation.sh
```

The script refuses to overwrite an existing run. Its primary outputs are:

- `runs/he_lit/ground`: ground-state checkpoints and logs.
- `runs/he_lit/lit`: LIT checkpoints, logs, and `lit_spectrum.npz`.

Use `CUDA_VISIBLE_DEVICES` to select devices.

## Run the postprocessing

After the GPU calculation finishes:

```bash
contrib/lit/examples/he/scripts/reproduce_postprocessing.sh
```

This reads `runs/he_lit/lit/lit_spectrum.npz` and creates the inversion fit,
CSV tables, Markdown report, and PNG/PDF figures under `runs/he_lit/post`.
The inversion uses one pole, a constant background, ordinary unweighted least
squares, and the 0.750--0.830 Ha fit window.
