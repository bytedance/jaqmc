# Homogeneous electron gas

The `jaqmc electron-gas` command simulates the three-dimensional homogeneous
electron gas (HEG, or jellium) in a simple-cubic periodic cell. The
implementation follows the physical conventions of
[Li et al., Nature Communications 13, 7895 (2022)](https://doi.org/10.1038/s41467-022-35627-1)
and uses JaQMC's periodic wavefunction, Ewald energy, sampler, and training
workflow.

## Define the system

An HEG system is specified by the Wigner-Seitz radius `system.rs`, in Bohr, and
the spin populations `system.nspins = [n_up, n_down]`. For
$N=n_\uparrow+n_\downarrow$ electrons, JaQMC constructs a simple-cubic cell
whose volume and side length are

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
  system.nspins='[7,7]' \
  workflow.save_path=./runs/heg_n14_rs1
```

The default twist is Gamma. Set `system.twist` in fractional reciprocal-cell
coordinates to use a different twist:

```bash
jaqmc electron-gas train \
  system.rs=1.0 \
  system.nspins='[7,7]' \
  system.twist='[0.5,0.0,0.0]' \
  workflow.save_path=./runs/heg_n14_twist
```

Integer shifts of the twist are physically equivalent.

## How it works

The Hamiltonian contains the ordinary Euclidean kinetic energy and the
electron-electron Ewald interaction with a uniform neutralizing positive
background. There are no nuclei, electron-nucleus features, atomic envelopes,
or PySCF calculation.

Pretraining instead uses analytic plane-wave Slater determinants. Within each
spin channel, reciprocal lattice vectors are filled in increasing
$|\mathbf{k}|^2$ order with deterministic tie-breaking. A plane-wave momentum
has the form

$$
\mathbf{k} = (\mathbf{n}+\mathbf{t})\mathbf{B},
$$

where $\mathbf{n}$ is an integer triplet, $\mathbf{t}$ is the fractional twist,
and $\mathbf{B}$ is the reciprocal-cell matrix.

| Concept | Configuration | Convention |
| --- | --- | --- |
| Density | `system.rs` | Bohr |
| Spin sector | `system.nspins` | two non-negative counts, not both zero |
| Boundary condition | `system.twist` | fractional reciprocal coordinates |
| Positive background | automatic | charged Ewald correction |
| Pretraining target | automatic | occupied plane waves |

## Quick local check

This small CPU run checks configuration, pretraining, one KFAC step, and
checkpoint writing. It is a workflow sanity check, not an energy benchmark.

```bash
JAX_PLATFORMS=cpu jaqmc electron-gas train \
  system.rs=1.0 \
  system.nspins='[1,1]' \
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

Use `--dry-run` to inspect the resolved configuration without starting
pretraining or VMC:

```bash
jaqmc electron-gas train \
  system.rs=1.0 \
  system.nspins='[7,7]' \
  workflow.save_path=./runs/heg_dry_run \
  --dry-run
```

## Kinetic energy

The standard kinetic-energy implementation is selected through
`estimators.energy.kinetic.mode`. The forward-Laplacian implementation can be
enabled explicitly:

```bash
jaqmc electron-gas train \
  system.rs=1.0 \
  system.nspins='[7,7]' \
  estimators.energy.kinetic.mode=forward_laplacian \
  workflow.save_path=./runs/heg_forward_laplacian
```

`estimators.energy.kinetic.vmap_chunk_size` changes only the memory scheduling;
it does not change the statistical batch size.

## Evaluation

Evaluation restores a trained wavefunction and samples it without parameter
updates. The system and wavefunction settings must match the training run.

```bash
jaqmc electron-gas evaluate \
  system.rs=1.0 \
  system.nspins='[7,7]' \
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
pretraining, and the neutralizing-background Ewald energy.

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
