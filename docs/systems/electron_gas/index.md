# Homogeneous electron gas

The `jaqmc electron-gas` command simulates the three-dimensional homogeneous
electron gas (HEG, or jellium) in a simple-cubic periodic cell. The implementation
follows the physical conventions of [Li et al., Nature Communications 13,
7895 (2022)](https://doi.org/10.1038/s41467-022-35627-1) and reuses JaQMC's
periodic wavefunction, Ewald, sampler, and workflow components. The public
[DeepSolid repository](https://github.com/bytedance/DeepSolid) provides the
historical implementation.

## Physical convention

The user specifies the Wigner-Seitz radius `system.rs` in Bohr and the spin
populations `system.nspins = [n_up, n_down]`. For
$N=n_\uparrow+n_\downarrow$, JaQMC constructs a simple-cubic cell with

$$
\Omega = \frac{4\pi}{3}N r_s^3, \qquad L=\Omega^{1/3}.
$$

The Hamiltonian contains the ordinary Euclidean kinetic energy and the
electron-electron Ewald energy with a uniform neutralizing positive background.
There are no nuclei, electron-nucleus features, atomic envelopes, or PySCF
calculation. Instead, pretraining uses analytic plane-wave Slater determinants.
Within each spin channel, reciprocal lattice vectors are filled in increasing
$|\mathbf{k}|^2$ order with deterministic tie-breaking.

`system.twist` is expressed in fractional reciprocal-cell coordinates and
defaults to Gamma. Integer shifts of the twist are equivalent.

| Concept | JaQMC representation | Invariant |
| --- | --- | --- |
| Density | `system.rs` in Bohr | $\Omega=4\pi N r_s^3/3$ |
| Spin sector | `system.nspins` | Two non-negative counts, not both zero |
| Plane waves | analytic pretraining reference | $(\mathbf{n}+\mathbf{t})\mathbf{B}$ |
| Positive background | charged Ewald correction | simple-cubic Madelung limit |
| HEG ansatz | periodic network with empty atom arrays | null atomic envelope |

## Quick smoke test

The small checked-in configuration exercises pretraining, KFAC training, and
checkpoint writing without claiming a converged energy:

```bash
jaqmc electron-gas train --yml docs/systems/electron_gas/cpu_smoke.yml

jaqmc electron-gas evaluate \
  system.rs=1.0 system.nspins='[1,1]' \
  wf.hidden_dims_single='[8]' wf.hidden_dims_double='[4]' wf.ndets=1 \
  jax.enable_x64=true workflow.batch_size=8 \
  workflow.source_path=./runs/heg_cpu_smoke \
  workflow.save_path=./runs/heg_cpu_smoke_eval \
  run.iterations=2 run.burn_in=1
```

This is a workflow sanity check, not a unit test or an energy benchmark. A
successful run stays finite, restores the training checkpoint, and writes
`evaluation_stats.h5` and `evaluation_digest.npz`.

## Reproducible GPU configurations

The repository provides two production-scale starting points:

| Purpose | Training | Evaluation |
| --- | --- | --- |
| 14-electron $r_s=1$ validation | [rs1_n14_gpu.yml](rs1_n14_gpu.yml) | [rs1_n14_gpu_eval.yml](rs1_n14_gpu_eval.yml) |
| 54-electron Fig. 2h protocol | [fig2h_n54_train.yml](fig2h_n54_train.yml) | [fig2h_n54_eval.yml](fig2h_n54_eval.yml) |

Inspect a configuration without launching work:

```bash
jaqmc electron-gas train \
  --yml docs/systems/electron_gas/fig2h_n54_train.yml --dry-run
jaqmc electron-gas evaluate \
  --yml docs/systems/electron_gas/fig2h_n54_eval.yml --dry-run
```

Always override `workflow.save_path` and `workflow.source_path` with persistent,
density-specific directories for production runs.

### Li et al. Fig. 2h protocol

Figure 2h reports the correlation error for a 54-electron closed-shell HEG in a
simple-cubic cell at

$$
r_s=0.5,\ 1,\ 2,\ 5,\ 10,\ 20\ \mathrm{Bohr}.
$$

The paper and its Supplementary Tables 1, 2, and 14 specify one determinant,
three `(256, 32)` periodic-network layers, a global batch of 4096, 1,000
pretraining steps, 300,000 KFAC training steps, Float64 precision, and 50,000
fixed-wavefunction inference steps. The published 54-electron runs used 32 A100
GPUs.

The JaQMC configurations make three implementation choices explicit:

- `distance_type: tri`, the current smooth JaQMC periodic feature;
- `full_det: true`, the current JaQMC solid determinant parameterization; and
- `forward_laplacian` for training and evaluation kinetic energies.

Consequently, this is a reproduction of the physical system, statistical
protocol, and comparison observable, not a bit-for-bit replay of DeepSolid's
network implementation. In particular, the paper writes the wavefunction as
spin-factorized determinants, while the current JaQMC solid ansatz uses a full
determinant.

Each density is an independent train/evaluate pair:

```bash
RS=1
ROOT=./runs/fig2h_n54

jaqmc electron-gas train \
  --yml docs/systems/electron_gas/fig2h_n54_train.yml \
  system.rs=${RS} workflow.save_path=${ROOT}/rs${RS}/train

jaqmc electron-gas evaluate \
  --yml docs/systems/electron_gas/fig2h_n54_eval.yml \
  system.rs=${RS} \
  workflow.source_path=${ROOT}/rs${RS}/train \
  workflow.save_path=${ROOT}/rs${RS}/evaluation
```

A shorter 20,000-step run at every density is useful as a finite/trainable gate
before committing to the full campaign. These gate energies are diagnostics,
not converged Fig. 2h points and must not be mixed with the 50,000-step
fixed-wavefunction evaluations.

### Analysis

Figure 2h plots

$$
100\left[1-\frac{E-E_{HF}}{E_{BF-DMC}-E_{HF}}\right].
$$

All energies are per electron. JaQMC records the 54-electron cell energy, so the
blocked mean and standard error are divided by 54 before comparison. Reblock the
saved evaluation series; do not use the uncorrelated-sample standard error.

The [machine-readable reference data](fig2h_reference.json) contain the values
from Supplementary Table 14 and their sources. One provenance subtlety is
preserved explicitly: the bars labelled `DCD` in the published figure match the
TC-DCD values in Liao et al. Table II, rather than the ordinary-DCD values copied
into Supplementary Table 14.

After all six evaluations have exactly 50,000 finite rows, generate the JaQMC
overlay input with:

```bash
uv run --group analysis python -m jaqmc.app.electron_gas.fig2h_analysis \
  --reference docs/systems/electron_gas/fig2h_reference.json \
  --evaluation 0.5=/path/rs0.5/evaluation/evaluation_stats.h5 \
  --evaluation 1=/path/rs1/evaluation/evaluation_stats.h5 \
  --evaluation 2=/path/rs2/evaluation/evaluation_stats.h5 \
  --evaluation 5=/path/rs5/evaluation/evaluation_stats.h5 \
  --evaluation 10=/path/rs10/evaluation/evaluation_stats.h5 \
  --evaluation 20=/path/rs20/evaluation/evaluation_stats.h5 \
  --output fig2h_jaqmc_results.json

uv run --group analysis python docs/systems/electron_gas/plot_fig2h.py \
  --jaqmc-results fig2h_jaqmc_results.json \
  --output fig2h_jaqmc.png
```

The analyzer rejects missing densities, unexpected row counts, non-finite
energies, and series without a pyblock optimum. It records the HDF5 checksum,
blocked uncertainty, imaginary-energy diagnostics, and difference from the
paper's Net result at every density.

## Scope

The focused tests cover the cell-volume relation, occupied closed shells,
twisted boundary phase, uniform walker initialization, neutralizing-background
Ewald constant, Float64 parameter initialization, and a finite forward
Laplacian. Training convergence remains a GPU validation result and is not
encoded as a unit test.

The current scope excludes twist averaging, structure-factor corrections,
thermodynamic-limit extrapolation, and HEG-specific alternative neural
architectures.

## Configuration reference

```{eval-rst}
.. config-context::
   :preset: jaqmc.app.electron_gas.workflow.ElectronGasTrainWorkflow.default_preset
```

```{eval-rst}
.. config-defaults:: jaqmc.app.electron_gas.config.ElectronGasConfig
   :prefix: system

.. config-defaults:: jaqmc.app.electron_gas.wavefunction.ElectronGasWavefunction
   :prefix: wf
```

Shared workflow, sampler, optimizer, writer, and runtime options are documented
under [Running Workflows](../../guide/running-workflows.md) and
[Runtime Configuration](../../guide/runtime-configuration.md).
