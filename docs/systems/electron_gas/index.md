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

In atomic units, JaQMC uses the finite-cell jellium Hamiltonian

$$
\begin{aligned}
\hat H ={}&
-\frac{1}{2}\sum_{i=1}^{N}\nabla_i^2
+\frac{1}{2}\sum_{i,j=1}^{N}\sum_{\mathbf R}^{\prime}
\frac{\operatorname{erfc}\!\left(\alpha
\lvert\mathbf r_i-\mathbf r_j+\mathbf R\rvert\right)}
{\lvert\mathbf r_i-\mathbf r_j+\mathbf R\rvert} \\
&+\frac{2\pi}{\Omega}\sum_{\mathbf G\ne 0}
\frac{e^{-G^2/(4\alpha^2)}}{G^2}
\left|\sum_{i=1}^{N}e^{i\mathbf G\cdot\mathbf r_i}\right|^2
-\frac{\alpha N}{\sqrt{\pi}}
-\frac{\pi N^2}{2\alpha^2\Omega}.
\end{aligned}
$$

Here $\mathbf R$ and $\mathbf G$ are direct- and reciprocal-lattice vectors,
respectively, and the prime omits $i=j,\mathbf R=0$. The final term is the
uniform neutralizing-background correction; the Ewald parameter $\alpha$
cancels when the sums are converged. The shared kinetic estimator defaults to
the sparse Forward Laplacian implementation.

The wavefunction uses JaQMC's periodic solid ansatz without envelope decay.
Each determinant is a full $N\times N$ electron-by-orbital matrix;
`wf.full_det=false` is not supported.

Pretraining uses analytic plane-wave orbitals rather than a self-consistent
field calculation. Within each spin channel, reciprocal lattice vectors are
filled in increasing $|\mathbf{k}|^2$ order. A deterministic tie-break selects
one occupation when the highest occupied shell is degenerate.

## Limitations

The current implementation supports three-dimensional simple-cubic finite
cells with two spin channels. Support for two-dimensional electron gases and
non-cubic simulation cells is planned future work. Individual twists are
supported, but twist averaging, finite-size corrections, and
thermodynamic-limit extrapolation must be performed separately.

## Configuration reference

The `system.*` options below are specific to the electron gas. The `wf.*`
options are shared with JaQMC's [periodic solid wavefunction](../solid/index.md);
the electron-gas workflow disables envelope decay and requires
`wf.full_det=true`.

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
