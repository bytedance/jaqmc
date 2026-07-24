# Homogeneous electron gas

The `jaqmc electron-gas` command simulates the three-dimensional homogeneous
electron gas (HEG, or jellium) in a simple-cubic periodic cell. It implements
the finite-cell jellium Hamiltonian and Wigner-Seitz density convention used in
[Li et al., Nature Communications 13, 7895 (2022)](https://doi.org/10.1038/s41467-022-35627-1)
using JaQMC's periodic wavefunction, Ewald energy, sampler, and training
workflow. It is not a line-by-line port of
[DeepSolid](https://github.com/bytedance/DeepSolid): the neural ansatz,
determinant layout, and training defaults follow JaQMC.

## Define the system

An HEG system is specified by the Wigner-Seitz radius `system.rs`, the total
electron count `system.nelectrons`, and the spin projection `system.s_z`. These
three fields are required; JaQMC does not assume a default physical system. It
derives the spin populations from

$$
n_\uparrow + n_\downarrow = N,
\qquad
n_\uparrow - n_\downarrow = 2s_z.
$$

For $N$ electrons, JaQMC constructs a simple-cubic cell whose volume and side
length are

$$
\Omega = \frac{4\pi}{3}N r_s^3,
\qquad
L = \Omega^{1/3}.
$$

For example, the following command trains a 14-electron, unpolarized HEG at
$r_s=1$:

```bash
jaqmc electron-gas train \
  system.rs=1.0 \
  system.nelectrons=14 \
  system.s_z=0 \
  workflow.save_path=./runs/heg_n14_rs1
```

The default twist is Gamma. Set `system.twist`, in fractional reciprocal-cell
coordinates, to use a different twist. Integer shifts are physically
equivalent.

## How it works

The Hamiltonian contains the ordinary Euclidean kinetic energy and the
electron-electron Ewald interaction with a uniform neutralizing positive
background. The shared kinetic estimator defaults to the sparse Forward
Laplacian implementation on supported JAX versions. There are no nuclei,
electron-nucleus features, atomic envelopes, or PySCF calculation. Each
determinant channel uses one full $N\times N$ electron-by-orbital matrix.
Spin-block determinants are not implemented, and `wf.full_det=false` is
rejected.

Pretraining instead uses analytic plane-wave Slater determinants. Within each
spin channel, reciprocal lattice vectors are filled in increasing
$|\mathbf{k}|^2$ order with deterministic tie-breaking. A plane-wave momentum
has the form

$$
\mathbf{k} = (\mathbf{n}+\mathbf{t})\mathbf{B},
$$

where $\mathbf{n}$ is an integer triplet, $\mathbf{t}$ is the fractional twist,
and $\mathbf{B}$ is the reciprocal-cell matrix.

This is a noninteracting free-electron occupancy, not a self-consistent
Hartree-Fock calculation. For a partially filled degenerate shell, the
deterministic tie-break selects one representative occupation.

## Quick local check

This small CPU run checks configuration, pretraining, one KFAC step, and
checkpoint writing. It is a workflow sanity check, not an energy benchmark.

```bash
JAX_PLATFORMS=cpu jaqmc electron-gas train \
  system.rs=1.0 \
  system.nelectrons=2 \
  system.s_z=0 \
  workflow.batch_size=8 \
  workflow.save_path=./runs/heg_smoke \
  wf.hidden_dims_single='[8]' \
  wf.hidden_dims_double='[4]' \
  wf.ndets=1 \
  sampler.steps=1 \
  pretrain.run.iterations=1 \
  pretrain.run.burn_in=1 \
  train.run.iterations=1 \
  train.run.burn_in=1
```

## Evaluation

Evaluation restores a trained wavefunction and samples it without parameter
updates. The system and wavefunction settings must match the training run.

```bash
jaqmc electron-gas evaluate \
  system.rs=1.0 \
  system.nelectrons=14 \
  system.s_z=0 \
  workflow.source_path=./runs/heg_n14_rs1 \
  workflow.save_path=./runs/heg_n14_rs1_eval
```

For statistically meaningful energies, reblock the saved
`evaluation_stats.h5` series instead of treating consecutive samples as
independent. See [Running Workflows](../../guide/running-workflows.md) for
checkpoint and evaluation mechanics.

## Scope

The current implementation supports a three-dimensional simple-cubic cell,
two spin channels, Gamma or twisted boundary conditions, plane-wave
pretraining, full determinants, and the neutralizing-background Ewald energy.

It does not yet include twist averaging, structure-factor corrections,
thermodynamic-limit extrapolation, or HEG-specific alternative neural
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
