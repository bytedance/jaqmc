# Troubleshooting

This page covers the most common issues you may encounter when running JaQMC simulations, along with their causes and fixes.

## Training Diverges / NaN in Energy

**Symptom:** Training stops abruptly, and the last logged step shows `energy=nan`.

**Cause:** Gradient explosion or numerical instability — common with Adam on
systems that need KFAC, with too-large learning rates, or with insufficient
pretraining.

**Fixes:**
- Lower the learning rate: `train.optim.learning_rate.rate=0.001`.
- Increase pretraining iterations to give the wavefunction a better starting
  point: `pretrain.run.iterations=5000`.
- Add burn-in to let the sampler equilibrate before gradient updates:
  `train.run.burn_in=200`.


## Energy Not Converging / High Variance

**Symptom:** `total_energy` fluctuates widely and does not plateau even after
many thousands of steps.

**Cause:** Insufficient pretraining, too-small batch size, too-high learning
rate, or too-small wavefunction ansatz.

**Fixes:**
- Increase pretraining: molecules typically need 1,000–10,000 pretrain steps.
- Increase `workflow.batch_size` (more walkers = lower variance per step).
- Try a larger network: increase `hidden_dims_single` or switch to Psiformer
  for molecules with more than ~30 electrons.
- Increase `ndets` (number of determinants) — the docs recommend 16 for
  production runs.
- See the production settings tables in the <project:../systems/molecule/index.md> and
  <project:../systems/solid/index.md> guides for system-specific recommendations.


## `pmove` Too High or Too Low

**Symptom:** The MCMC acceptance rate (`pmove`) stays far from the target
range of 0.50–0.55.

**What `pmove` means:** The fraction of proposed electron moves accepted by
the Metropolis-Hastings criterion, averaged over all walkers and sub-steps.

**If `pmove` is very high (> 0.8):** The step width is too small. The
adaptive mechanism will correct this automatically over a few hundred steps.
This is normal and expected at the start of a run.

**If `pmove` is very low (< 0.1):** Walkers are stuck — the wavefunction
landscape is extremely peaked. This often happens after bad initialization.
The adaptive mechanism will shrink the step width, but you can also:
- Lower sampler proposal width:
  - training: `train.sampler.initial_width=...`
  - evaluation: `sampler.initial_width=...`
- Add a burn-in period: `train.run.burn_in=500`.
- Increase pretraining.

## Shape Mismatch in Custom Components

**Symptom:** Errors involving unexpected coordinate shapes, failed concatenation,
or `vmap` axis mismatches while adding custom wavefunctions/estimators/samplers.

**Cause:** Mixing up one-walker `Data` shapes with advanced `BatchedData`
metadata, or using a custom layout inconsistently across components.

**Fixes:**
- First sanity check: inside `Wavefunction.__call__` or a per-walker estimator,
  `data.electrons` should usually be one walker, not `(batch, ...)`.
- Read the default and advanced conventions in <project:/extending/runtime-data-conventions.md>.
- Validate field-level shapes (for example `batched_data.data.electrons`), not
  the `BatchedData` wrapper itself.
- Confirm that every field listed in `fields_with_batch` actually carries a
  leading walker axis.
- Remember that unbatched fields are often shared arrays such as `atoms` or
  `charges`, not necessarily scalars.
- For built-in-style layouts, use per-walker `(n_particles, ndim)` and batched
  `(batch, n_particles, ndim)` for particle fields such as `electrons`.
- If your custom layout differs, keep it internally consistent across sampler,
  wavefunction, estimators, and workflow wiring.


## Config Typo / "Stopping Due to Invalid Configs"

**Symptom:** The run aborts immediately with:
```
Stopping due to invalid configs specified.
```
A warning above lists the unrecognized key(s).

**Cause:** A config key was passed (via CLI or YAML) that the config system
never read — most commonly a typo. For example, `train.run.iteration` instead
of `train.run.iterations`.

**Fixes:**
- Check the spelling of the key.
- Use `--dry-run` to see the full resolved config and verify all keys.
- Add `workflow.config.verbose=true` to see available fields and their
  descriptions.

```{tip}
CLI overrides must not have spaces around `=`. Write `key=value`, not
`key = value`.
```


## Spin/Electron Count Parity Error

**Symptom:**
```
ValueError: Total electrons (N) and spin (S) must have the same parity.
```

**Cause:** The `spin` parameter (number of unpaired electrons) must have the
same parity as the total electron count. For example, a 10-electron system
cannot have `spin=1`.

**Fix:** For a neutral molecule, count the total electrons and choose a `spin`
with matching parity: 0 for singlet, 2 for triplet, etc. For atoms, JaQMC
determines spin automatically for main-group elements. For transition metals,
specify `spin` explicitly.


## Checkpoint Resume Ends Immediately

**Symptom:** After restoring a checkpoint, the run terminates immediately
without any training steps.

**Cause:** The checkpoint was saved at step N and `train.run.iterations` is
still set to N. The loop condition `step < iterations` is false from the
start.

**Fix:** Increase `train.run.iterations` beyond the checkpoint step. For
example, if restoring at step 1000, set `train.run.iterations=2000`.

See <project:running-workflows.md> for more on resuming and checkpointing.


## No Checkpoint Found (Fresh Start)

**Symptom:** Log shows `No checkpoint to restore in: <path>`.

**This is not an error.** On the first run there is no checkpoint yet, and
training starts from step 0. The message is purely informational.


## Batch Size Not Divisible by Device Count

**Symptom:**
```
ValueError: Batch size N must be divisible by number of processes P.
```

**Fix:** Choose a `workflow.batch_size` that divides evenly by the number of
GPUs (or simulated CPU devices). For example, with 4 GPUs use batch sizes
like 4096 or 8192.


## Psiformer Not Available for Solids

**Symptom:** Cannot find or configure Psiformer for a solid simulation.

**Cause:** The Psiformer architecture is currently only available for molecule
simulations. Solid simulations use FermiNet extended with PBC features.

**Fix:** Use FermiNet (the default) for solid simulations.


## Transition Metal Spin Not Detected

**Symptom:**
```
NotImplementedError: Spin configuration for transition metals not set.
```

**Cause:** The automatic spin detection (`atom_config`) only handles
main-group elements (groups 1, 2, 13–18).

**Fix:** Specify the `spin` parameter explicitly in your system config rather
than relying on automatic detection.
