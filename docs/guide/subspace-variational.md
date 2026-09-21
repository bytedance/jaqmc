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

The optimizer stays under the native `train.optim` namespace. Adam is the
correctness-first subspace default; KFAC can be selected explicitly after the
determinant-state curvature path has been validated for the target network.

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
    vmap_chunk_size: 1
    pair_chunk_size: 4
    matrix_dtype: complex128

  diagnostics:
    condition_warning: 1.0e10
    solve_residual_warning: 1.0e-6
    max_imag_eigenvalue_warning: 1.0e-6

train:
  optim:
    module: jaqmc.optimizer.optax:adam
    learning_rate: 0.0001
  grads:
    vmap_chunk_size: 1
    clip_method: mad
    clip_scale: 5.0
```

The three chunk controls apply to different axes:

- `subspace.evaluation.vmap_chunk_size` controls determinant walkers entering
  the Rayleigh/local-energy estimator concurrently.
- `subspace.evaluation.pair_chunk_size` controls concurrent $(r,s)$ energy
  pairs inside one determinant walker.
- `train.grads.vmap_chunk_size` controls determinant walkers entering the
  streaming log-determinant gradient calculation.

Parameters and replica data remain in their original $M$-sized containers.
The streaming gradient estimator reduces each chunk immediately to parameter
PyTree sufficient statistics instead of reconstructing a
`[walkers, ...parameters]` tree. State-independent potential energy is
evaluated $M$ times and reused across state columns. Walker sharding remains
the existing JaQMC behavior; every device keeps the complete state axis.

This is intentionally walker-axis data parallelism. With a global determinant
batch of $B$ and $N$ devices, each device receives approximately $B/N$ walkers
while retaining all $M$ component states. Pair and gradient chunk sizes control
single-device concurrency and memory; they do not shard the state axis.

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
gradient uses the full complex $\operatorname{Tr}R_L$ through
`StreamingLossAndGrad`. The workflow forces this loss key while preserving
`train.grads` configuration for chunking and clipping. No NetKet runtime or
second Hamiltonian implementation is used.

## Grassmann optimization

No separate Grassmann wavefunction or QGT implementation is required. For the
determinant state,

$$
\partial_\mu\log\det\Phi
=\operatorname{Tr}(\Phi^{-1}\partial_\mu\Phi),
$$

so passing `DeterminantStateWavefunction.logpsi` to JaQMC's existing
`SROptimizer` produces the Grassmann score and metric. The recommended
production preset is `configs/workflows/subspace_grassmann_sr.yml`; it retains
JaQMC's chunking, mixed-precision solve, SPRING, and multi-device reductions
while disabling robustness extensions for a controlled first comparison.

For small, single-device A/B checks,
`configs/workflows/subspace_gvmc_reference_sr.yml` selects
`GVMCReferenceSROptimizer`. This backend independently implements the published
GVMC sample-space minSR and Kaczmarz/SPRING equations from the
[official accompanying repository](https://github.com/cqsl/GVMC) behind
JaQMC's unchanged `OptimizerLike` interface. JaQMC supplies the native VMC
gradient $2\operatorname{Re}(J^\dagger B)$; the reference adapter converts it
to the GVMC convention $\operatorname{Re}(J^\dagger B)$ before minSR and
stores its Kaczmarz direction in that convention. This conversion intentionally
does not change the shared gradient estimator or compensate through the learning
rate. The configured learning rate is applied directly as $-\eta\delta$, matching
the actual parameter update in `cqsl/GVMC`. Its source computes an auxiliary
normalized displacement with $\eta/\sqrt M$, but does not apply that quantity
to parameters. It is a numerical oracle, not a production backend.
Both presets set `train.grads.clip_method: none`, matching the unclipped force
used for the reference algorithm.

For a controlled plain-SR comparison, JaQMC native SR receives the factor-two
gradient directly, so `learning_rate_native` is approximately
`learning_rate_gvmc / 2`. This is only an initial A/B scale convention; robust
SR, adaptive damping, MARCH, or norm clipping need separate comparisons.

The Rayleigh estimator also reports the Grassmann Hamiltonian variance without
another Hamiltonian evaluation:

$$
\Sigma_H=\mathbb E[R_LE_L^*]-\mathbb E[R_L]\mathbb E[E_L]^*,\qquad
\operatorname{Var}_V(H)=\frac1M\operatorname{Re}\operatorname{Tr}\Sigma_H.
$$

The appended writer fields are `grassmann_average_energy`,
`grassmann_hamiltonian_variance`,
`grassmann_hamiltonian_variance_matrix`, and `grassmann_hamiltonian_std`.
Existing Rayleigh and subspace-energy fields are unchanged. In particular,
`grassmann_hamiltonian_variance` diagnoses invariance of the entire span and is
not interchangeable with `subspace_energy_var` or elementwise
`local_rayleigh_variance`.

## Diagnostics

Monitor `grassmann_hamiltonian_variance`, `amplitude_sigma_min`, `amplitude_condition`,
`rayleigh_solve_residual`, `max_ritz_imag`, and their warning fields.  A large
condition number indicates nearly dependent component states, but is diagnostic
only and does not remove a sample from the Monte Carlo measure. Every finite
input attempts the Rayleigh solve. `rayleigh_valid` becomes false only for a
catastrophic numerical failure such as non-finite input, solution, or residual;
then the optimizer update is skipped before it can change parameters or
optimizer state, and training aborts through the native checkpoint path. Normal
gradients use every determinant sample without masked averaging. Training logs
also include `grad_norm` and `update_norm`; an invalid gated step reports a zero
update norm.

## Hydrogen example

The public example
[`examples/atoms/hydrogen_subspace.yml`](../../examples/atoms/hydrogen_subspace.yml)
targets the $M=5$ span containing the hydrogen 1s state and fourfold $n=2$
manifold. Its exact sorted Ritz spectrum is
$[-0.5,-0.125,-0.125,-0.125,-0.125]$ Ha, with Ky-Fan trace $-1.0$ Ha.

Run the example through the molecule workflow:

```console
jaqmc molecule subspace-train \
  --yml examples/atoms/hydrogen_subspace.yml
```

The same core feature is available to periodic systems through
`jaqmc solid subspace-train`; system-specific research and hardware overlays
are intentionally kept outside this core contribution.

The current sampler is the correctness-first full-recompute implementation: one
replica row moves per proposal while the determinant is reevaluated through the
normal sample-plan log-probability callback.  A cached Sherman--Morrison backend
can be added later behind the same sampler/wavefunction interfaces.
