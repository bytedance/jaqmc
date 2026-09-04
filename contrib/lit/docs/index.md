# LIT spectra

Electric-dipole Lorentz integral transform (LIT) calculations are provided by
the optional `jaqmc-contrib-lit` contribution. Installing it adds a top-level
`lit` command to JaQMC; the normal molecule training, evaluation, sampler, and
checkpoint code paths do not import LIT. This package is experimental and is
maintained independently from the JaQMC core API.

Install from a source checkout with:

~~~bash
uv sync --package jaqmc-contrib-lit
source .venv/bin/activate
~~~

After installation, jaqmc --help includes:

~~~text
lit run       Compute a molecular electric-dipole LIT spectrum.
lit invert    Fit a saved raw LIT archive on the CPU.
~~~

Without `jaqmc-contrib-lit`, these commands are absent and the rest of JaQMC
remains usable.

## Run a spectrum

First train the ground-state wavefunction with the standard molecule workflow:

~~~bash
jaqmc molecule train --yml ground.yml \
  workflow.save_path=./runs/ground
~~~

Then run LIT using that checkpoint:

~~~bash
jaqmc lit run --yml lit.yml \
  workflow.save_path=./runs/lit \
  lit.ground.checkpoint_path=./runs/ground
~~~

The LIT settings are grouped by responsibility. A compact configuration looks
like:

~~~yaml
lit:
  eta: 0.003
  axes: x
  output_filename: lit_spectrum.npz

  omega:
    minimum: 0.750
    maximum: 0.900
    points: 601

  ground:
    checkpoint_path: ./runs/ground
    allow_untrained: false
    energy: null
    energy_steps: 32
    burn_in: 150

  source:
    center_steps: 64
    center_override: [0.0, 0.0, 0.0]
    norm_override: null
    burn_in: 150
    floor: 0.0001
    train_pool_batches: 512
    eval_pool_batches: 64
    pool_stride: 4
    reuse_pool: true
    save_pool: true
    distillation_iterations: 3000

  parallel:
    mode: local_devices
    train_batch_size_per_device: 1024
    eval_batch_size_per_device: 1024

  solver:
    iterations: 6000
    learning_rate: 0.002
    reverse_kl_weight: 1.0
    spring_epsilon: 0.001
    spring_decay: 0.99
    spring_damping_floor: 1.0e-12
    sr_max_norm: 0.02
    warm_start_omega: -3.674932217565499
    warm_start_iterations: 1500
    selection_interval: 100
    log_interval: 100

  continuation:
    iterations: 6000
    step_fraction: 0.2
    step_growth_factor: 1.25
    fidelity_retention: 0.95
    ess_fraction_minimum: 0.05
    minimum_step: null
    maximum_points: 1000

  ansatz:
    determinants: 16
    hidden_dims_single: [64, 64, 64, 64]
    hidden_dims_double: [16, 16, 16, 16]
    envelope: abs_isotropic
    orbitals_spin_split: true
~~~

Use omega.values for an explicit strictly increasing grid instead of the
minimum/maximum/points grid.

The complete He input example is in `contrib/lit/examples/he`. It includes the
ground-state, LIT, and inversion configurations plus executable calculation and
postprocessing scripts. Raw spectra, logs, fits, tables, reports, figures,
checkpoints, and source pools are generated under `runs/he_lit` and are not
included in the package source.

## Output and execution model

lit_spectrum.npz contains the frequency grid, the signed LIT, the finite-width
broadened diagnostic, held-out fidelity and reverse-KL values,
importance-sampling diagnostics, error monitors, continuation records, and
matched jackknife pseudo-values.

The reported frequency scan is serial because each response state initializes
the next frequency. parallel.mode: local_devices only shards the walkers for
one frequency across devices controlled by one JAX process; it does not split
the continuation chain. Configured global batches must be divisible by the
local device count. Alternatively, per-device batch sizes are multiplied by
that count.

The current source-sector policy is deliberately narrow:

- one-center atoms use an automatically diagnosed hard response parity
  opposite to the restored ground state;
- multi-center systems are accepted only when sector discovery returns C1.

The response ansatz is independent of the ground-state ansatz. LIT-specific
source pools, continuation checkpoints, and corrupt-checkpoint fallback live
inside jaqmc-contrib-lit; they do not alter JaQMC's standard checkpoint format.

## Fit saved LIT data

The GPU workflow stops after writing the raw archive. Line count, starting
positions, bounds, and background order are explicit postprocessing
hypotheses. For example:

~~~yaml
inversion:
  input_paths: [runs/lit/lit_spectrum.npz]
  output_path: runs/lit/line_fit_k1_b0.npz
  pole_count: 1
  background_order: 0
  require_determined: true
~~~

~~~bash
jaqmc lit invert --yml invert.yml
~~~

The fitted model is

$$
\mathcal L_a(\omega,\eta)
=\sum_n\frac{I_{an}}{(\omega_n-\omega)^2+\eta^2}
+\sum_{j=0}^{m}b_{aj}x(\omega)^j.
$$

Line centers are shared across response axes; transition strengths and
background coefficients are axis-specific. The fit accepts one fixed
broadening width. For all three Cartesian axes, the output also reports

$$
f_{0n}=\frac{2}{3}\omega_n(I_{xn}+I_{yn}+I_{zn}).
$$

The output path must differ from every raw input path, so postprocessing cannot
overwrite the expensive LIT result.
