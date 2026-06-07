# Troubleshooting

When you debug a run, keep the evidence reproducible. Save the original command
or YAML, the resolved `config.yaml`, the seed, and the first bad step. For
final energy conclusions, prefer a separate evaluation run over the training
log; see
<project:analyzing-evaluations.md>.

## When Training Starts but Metrics Look Wrong

### Training Diverges or Energy Becomes NaN

By default, VMC training stops when `loss` becomes NaN
(`train.run.stop_on_nan="loss"`). The rows just before the failure in
`train_stats.csv` or other training outputs usually contain the best clues.

- If the bad row is step 0, look at initialization, pretraining quality, system
  setup, or the local-energy estimator.
- If `pmove` was already near 0 or 1 before the NaN, start with sampling:
  proposal width, burn-in, and whether the initialized wavefunction gives a
  reasonable distribution.
- If `energy:kinetic`, `energy:potential`, or another `energy:*`
  component becomes NaN before `loss`, debug that estimator or the wavefunction
  values it differentiates.
- If the previous row was finite and the next row becomes NaN with a large jump
  in `total_energy` or `total_energy_var`, the previous optimizer update likely
  pushed the parameters into an unstable region. Reduce the update size: lower
  `train.optim.learning_rate.rate`; for KFAC, also consider a smaller
  `train.optim.norm_constraint` or a larger `train.optim.damping`.
- If only rare rows spike while most rows are finite, treat it as a local-energy
  tail problem rather than a generic optimizer failure. See
  [Rare Spikes or Heavy Energy Tails](#rare-spikes-or-heavy-energy-tails).

If this started after adding custom code, check shapes early. Custom component
shape bugs often show up first as NaNs or other errors; see
[Shape Mismatch in Custom Components](#shape-mismatch-in-custom-components).

### Training Looks Good but Evaluation Gets Worse

**Symptom:** the training log reports a low or improving `total_energy`, but a
separate evaluation from the checkpoint gives a higher energy or larger error
bar.

This usually means the training-time estimate is flattering the run. Common
causes are sampling bias during optimization, train/eval configuration drift,
too little evaluation burn-in, or comparing different observables.

Try these first:

- Compare the training and evaluation configs and make sure the system,
  wavefunction, and other relevant settings match.
- Increase evaluation burn-in if the sampler still looks far from equilibrium.
- If training `pmove` or energy estimates drift suspiciously, revisit sampler
  settings in training as well.

### Energy Not Converging or High Variance

**Symptom:** `total_energy` fluctuates widely, `total_energy_var` remains high,
or the curve does not plateau after many steps.

Likely causes are insufficient pretraining, too few walkers, an overly large
learning rate, poor sampler mixing, or a wavefunction that is too small for the
system.

Try these checks first:

- If `pmove` stays far from 0.5 for many steps, tune
  `train.sampler.initial_width` first. Persistent mismatch usually means the
  proposal scale is off.
- If your workflow includes pretraining, increase it. For molecular systems,
  1,000-10,000 pretraining steps is a common starting range, and a pretrain
  loss around `1e-4` is often reasonable.
- Tune the learning rate and other optimizer settings.
- Try a larger or more expressive wavefunction.

### Energy Is Suspiciously Low or Another Observable Looks Wrong

**Symptom:** the energy is below a trusted variational reference, or a change
looks "too good" without a matching reduction in variance. The same pattern can
show up for other observables as well.

- Re-evaluate the checkpoint with frozen parameters to separate a training-time
  sampling issue from a genuinely lower estimate.
- Check the system definition and make sure you are comparing the same physical
  system as the reference.
- If `pmove` stays far from 0.5 for many steps, fix
  `train.sampler.initial_width` before trusting the estimate.
- Inspect walker trajectories or checkpointed walker configurations. Walkers
  that stay spatially localized, break an expected symmetry, or get stuck in an
  unexpected spin arrangement usually indicate insufficient sampling. In that
  case, revisit burn-in, the proposal strategy, and walker initialization.
- Verify that the chosen wavefunction family enforces the intended
  antisymmetry and continuity.
- If a custom estimator contributes to `energy:*`, validate signs, factors,
  units, and batch reductions against a simple trusted case.

### Rare Spikes or Heavy Energy Tails

**Symptom:** most steps look normal, but occasional spikes dominate
`total_energy_var` or make evaluation error bars unstable.

**Likely causes:** cusp-condition problems, discontinuities, singular-point
handling, coordinate wrapping errors.

Start by finding where the high-energy walkers are. In particular, look for
spikes near nuclei, close electron-electron approaches, singular points, or
periodic cell boundaries.

## Sampling Symptoms

### `pmove` Too High or Too Low

**Symptom:** the MCMC acceptance rate (`pmove`) stays far from the healthy
range. The sampler adapts toward the default target range of `0.50-0.55`, while
values around `0.3-0.7` are usually acceptable.

**What `pmove` means:** the fraction of proposed electron moves accepted by the
Metropolis-Hastings rule, averaged over walkers and sampler sub-steps.

Short transients at the start of a run are normal. Persistent extremes are the
signal to act.

If `pmove` stays very high, for example above `0.8`, increase the starting
proposal width: `train.sampler.initial_width=...` for training or
`sampler.initial_width=...` for evaluation. If `pmove` stays very low, reduce
the width instead.

See <project:sampling.md> for the sampler parameters and adaptation behavior.

## Configuration and Runtime Errors

### Config Typo or "Stopping Due to Invalid Configs"

**Symptom:** the run aborts immediately with:

```text
Stopping due to invalid configs specified.
```

A warning above the error lists the keys that were specified but never read.
This usually means a typo, such as `train.run.iteration` instead of
`train.run.iterations`.

Try these fixes:

- Check the spelling of the key named in the warning.
- Use `--dry-run` to inspect the resolved config.
- Add `workflow.config.verbose=true` to show available fields and their
  descriptions.

```{tip}
CLI overrides must not have spaces around `=`. Write `key=value`, not
`key = value`.
```

### Spin/Electron Count Parity Error

**Symptom:**

```text
ValueError: Impossible s_z=... for N electrons.
```

**Cause:** `system.s_z` must be compatible with the total electron count. In
practice, `2 * s_z` must have the same parity as the total number of explicit
electrons. For example, a 10-electron system cannot have `s_z=0.5`.

**Fix:** for a neutral molecule, count the total electrons and choose a
compatible `s_z`: `0` for a singlet, `1` for a triplet, and so on. For
`system.module=atom`, JaQMC fills in the neutral atom's default `s_z`
automatically from the bundled element table. Override `system.s_z`
explicitly when you need a different charge or spin state.

### Checkpoint Resume Ends Immediately

**Symptom:** after restoring a checkpoint, the run terminates immediately without
any training steps.

**Cause:** the checkpoint was saved at step `N`, and `train.run.iterations` is
still set to `N`. The loop condition `step < iterations` is false from the
start.

**Fix:** increase `train.run.iterations` beyond the checkpoint step. For
example, if you restore at step 1000, set `train.run.iterations=2000`.

See <project:running-workflows.md> for more on resuming and checkpointing.

### No Checkpoint Found on a Fresh Training Run

**Symptom:** the log shows:

```text
No checkpoint to restore in: <path>
```

On the first training run, this is informational. There is no checkpoint yet, so
training starts from step 0.

### Batch Size Not Divisible by Process Count

**Symptom:**

```text
ValueError: Batch size N must be divisible by number of processes P.
```

**Fix:** choose a `workflow.batch_size` that divides evenly by the number of JAX
processes. For example, with 4 processes use a batch size like `4096` or `8192`.

## Custom Component Errors

### Shape Mismatch in Custom Components

**Symptom:** errors mention unexpected coordinate shapes, failed concatenation,
or `vmap` axis mismatches while adding custom wavefunctions, estimators, or
samplers.

**Likely causes:** mixing up one-walker `Data` shapes with `BatchedData`
metadata, or using a custom layout inconsistently across components.

Start with these checks:

- Inside `Wavefunction.__call__` or a per-walker estimator, `data.electrons`
  should usually represent one walker, not `(batch, ...)`.
- Read the runtime data conventions in
  <project:/extending/runtime-data-conventions.md>.
- Validate field-level shapes, such as `batched_data.data.electrons`, not only
  the `BatchedData` wrapper itself.
- Confirm that every field listed in `fields_with_batch` actually carries a
  leading walker axis.
- Remember that unbatched fields are often shared arrays such as `atoms` or
  `charges`, not necessarily scalars.
- For built-in-style layouts, use per-walker `(n_particles, ndim)` and batched
  `(batch, n_particles, ndim)` for particle fields such as `electrons`.
- If your custom layout differs, keep it internally consistent across sampler,
  wavefunction, estimators, and workflow wiring.

## When Asking for Help

Include this evidence pack in your issue or discussion:

- Primary symptom and first bad step.
- Exact command, YAML files, and resolved config.
- Seed, checkpoint path, and whether parameters were frozen for comparison.
- Training and evaluation values for `total_energy`, `total_energy_var`, and
  `pmove`.
- The cheapest separating test you already ran and what changed.
