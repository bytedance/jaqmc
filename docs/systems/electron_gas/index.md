# Electron gas

The `jaqmc electron-gas` command provides workflows for periodic electron-gas
systems. This page describes the currently supported three-dimensional
homogeneous electron gas (HEG, or jellium) in a simple-cubic cell. It uses the
finite-cell jellium Hamiltonian and Wigner-Seitz density convention of
[Li et al., Nature Communications 13, 7895 (2022)](https://doi.org/10.1038/s41467-022-35627-1).

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

The default twist is Gamma. Set `system.twist`, in fractional reciprocal-cell
coordinates, to use a different twist. Integer shifts are physically
equivalent.

## Train and evaluate

The following command trains a 14-electron, unpolarized HEG at $r_s=1$:

```bash
jaqmc electron-gas train \
  system.rs=1.0 \
  system.nelectrons=14 \
  system.s_z=0 \
  workflow.save_path=./runs/heg_n14_rs1
```

Evaluate the trained wavefunction using the same system settings:

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

## How it works

The Hamiltonian combines Euclidean kinetic energy with the electron-electron
Ewald interaction and a uniform neutralizing background. The shared kinetic
estimator defaults to the sparse Forward Laplacian implementation.

The wavefunction uses JaQMC's periodic solid ansatz without envelope decay.
Each determinant is a full $N\times N$ electron-by-orbital matrix;
`wf.full_det=false` is not supported.

Pretraining uses analytic plane-wave orbitals rather than a self-consistent
field calculation. Within each spin channel, reciprocal lattice vectors are
filled in increasing $|\mathbf{k}|^2$ order. A deterministic tie-break selects
one occupation when the highest occupied shell is degenerate.

## Limitations

The current implementation supports three-dimensional simple-cubic finite
cells with two spin channels. Individual twists are supported, but twist
averaging, finite-size corrections, and thermodynamic-limit extrapolation must
be performed separately.

## Configuration reference

The `system.*` options below are specific to the electron gas. The `wf.*`
options are shared with JaQMC's periodic solid wavefunction; the electron-gas
workflow disables envelope decay and requires `wf.full_det=true`.

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
