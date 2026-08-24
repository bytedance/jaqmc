# Variational subspaces

JaQMC can jointly optimize a low-energy subspace without changing the physical
FermiNet, LapNet, or Psiformer implementation.  The extension evaluates one
native ansatz architecture with independent parameter sets, samples the
determinant state, and minimizes the real trace of its local Rayleigh matrix.
For complex wavefunctions the gradient estimator retains the complete complex
local trace; the real and imaginary parts are split only for reporting.

Use a separate workflow so ordinary ground-state commands and configuration
remain unchanged:

```console
jaqmc molecule subspace-train --yml config.yaml
jaqmc solid subspace-train --yml config.yaml
```

The determinant walker stores coordinates as `[walkers, states, electrons, 3]`.
The state/replica axis is never flattened into the physical electron axis, so
Hamiltonian estimators do not introduce interactions between replicas.  Initial
replicas are independent samples produced by the app's existing data
initializer.

## Configuration

The optimizer stays under the native `train.optim` namespace.  KFAC remains the
production default; SR and Optax can be selected in the same way as for normal
VMC training.

```yaml
subspace:
  n_states: 2

  initialization:
    mode: random  # or checkpoints
    # checkpoints: [/path/to/state0, /path/to/state1]

  sampling:
    steps: 10
    initial_width: 0.02
    update_mode: full

  evaluation:
    pair_chunk_size: 4
    matrix_dtype: complex128

  diagnostics:
    condition_warning: 1.0e10
    condition_hard_limit: 1.0e14
    min_valid_fraction: 0.99
    solve_residual_warning: 1.0e-6
    max_imag_eigenvalue_warning: 1.0e-6

train:
  optim:
    module: jaqmc.optimizer.kfac:KFACOptimizer
```

`pair_chunk_size` limits peak memory in the $M^2$ cross-local-energy
evaluation. Parameters and replica data remain in their original $M$-sized
containers and are selected dynamically inside each chunk. State-independent
potential energy is evaluated $M$ times and reused across state columns.
Walker sharding remains the existing JaQMC behavior; every device keeps the
complete state axis.

With `initialization.mode: checkpoints`, provide exactly `n_states` native
JaQMC checkpoint files or directories. Parameter PyTree structure and leaf
shapes are checked before the states are stacked.

## Reused native components

For every replica $R_r$ and component state $s$, the estimator calls the
already configured physical JaQMC energy pipeline to obtain
$E_{L,s}(R_r)$.  It then forms

$$
\Phi^{(H)}_{rs}=\Phi_{rs}E_{L,s}(R_r),\qquad
R_L=\operatorname{solve}(\Phi,\Phi^{(H)}).
$$

The reported scalar energy is $\operatorname{Re}\operatorname{Tr}R_L$; the
gradient uses the full complex $\operatorname{Tr}R_L$ through the existing
`LossAndGrad`. No NetKet runtime or second Hamiltonian implementation is used.

## Diagnostics

Monitor `amplitude_sigma_min`, `amplitude_condition`,
`rayleigh_solve_residual`, `max_ritz_imag`, and their warning fields.  A large
condition number indicates nearly dependent component states.  Non-finite
Rayleigh matrices are reported explicitly rather than silently hidden. All
matrix elements, diagnostics, and gradients use one whole-walker validity
mask. A step below `min_valid_fraction` is rolled back and aborts through the
native checkpoint recovery path before invalid optimizer state is retained.

The current sampler is the correctness-first full-recompute implementation: one
replica row moves per proposal while the determinant is reevaluated through the
normal sample-plan log-probability callback.  A cached Sherman--Morrison backend
can be added later behind the same sampler/wavefunction interfaces.
