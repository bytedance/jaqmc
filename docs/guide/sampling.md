# Sampling

At each training step, JaQMC draws electron configurations from $|\psi|^2$ using Markov Chain Monte Carlo (MCMC). These samples are used to estimate the local energy and its gradient. The quality of these samples — how many, how independent, how well-equilibrated — directly affects training stability and convergence.

## Number of Walkers

The sampler maintains a set of independent MCMC chains called **walkers**. The number of walkers is set by `workflow.batch_size` (default 4096). This is a *global* count — it is divided evenly across processes in multi-host setups and must be divisible by the number of processes.

More walkers reduce the variance of the energy estimate at each step, but increase memory and computation. Fewer walkers are cheaper per step but noisier. For production runs, the default of 4096 is a reasonable starting point; smaller systems or quick tests can use fewer (e.g., 512–2048).

```bash
workflow.batch_size=2048
```

## How MCMC Sampling Works

For each walker, the sampler proposes a move — displacing all electrons simultaneously — evaluates the wavefunction at the new position, and accepts or rejects the move based on the Metropolis-Hastings criterion. The fraction of accepted moves is the **pmove** (acceptance rate).

The proposal distribution is a Gaussian centered on the current position. Its width controls the step size: too wide and most proposals land in low-probability regions (low pmove), too narrow and walkers explore slowly (high pmove but poor mixing). Different workflows use different proposal geometries (e.g., PBC-wrapped for solids, spherical for quantum Hall), but the proposal is selected automatically and the configuration parameters below work the same way across all workflows.

## Key Parameters

- **`steps`** (default 10) — Number of Metropolis-Hastings updates per training step. More steps reduce autocorrelation between successive samples but increase computation per step.
- **`initial_width`** (default 0.1) — Starting standard deviation of the Gaussian proposal. The sampler adapts this automatically during training.
- **`pmove_range`** (default `(0.5, 0.55)`) — Target acceptance rate range. The sampler scales the proposal width up or down to stay in this range.
- **`adapt_frequency`** (default 100) — How often (in training steps) the proposal width is adjusted. Every `adapt_frequency` steps, the sampler checks the mean acceptance rate: if it exceeds the upper bound of `pmove_range`, the width is scaled up by 1.1×; if it falls below the lower bound, the width is scaled down by 1.1×.

## Tuning the Sampler

### Acceptance Rate

A healthy acceptance rate is typically 0.3–0.7, with the optimal range around 0.5. The default `pmove_range` of `(0.5, 0.55)` works well for most systems. If you see pmove persistently outside this range at the start of training, adjust `initial_width`:

- **pmove too low** → `initial_width` is too large. Reduce it.
- **pmove too high** → `initial_width` is too small. Increase it.

The sampler adapts the width automatically, but a good starting value avoids a long adaptation transient. Larger systems with more electrons generally need a smaller `initial_width`.

### Number of Steps

The default `steps` value works for most systems. Increase it if you observe high autocorrelation in the energy estimates (energy variance doesn't decrease as expected). Decrease it if per-step training time is a bottleneck and pmove is healthy.

### Burn-in

Walkers start from random positions near the atoms, which are usually far from the equilibrium distribution $|\psi|^2$. Training on these early, unequilibrated samples produces noisy gradients. The `train.run.burn_in` setting (default 100) runs a number of sampling-only steps before training begins — walkers are moved but no gradients are computed. This gives the MCMC chains time to equilibrate and the adaptive width to stabilize. Each burn-in step performs `steps` Metropolis-Hastings updates, so the total burn-in MH updates is `burn_in * steps`.

```bash
train.run.burn_in=100  # 100 steps × 10 MH updates = 1000 MH updates total
```

Burn-in is generally recommended for production runs (e.g., 100–500 steps), and the training default is 100. For quick tests, you can set `train.run.burn_in=0` to start immediately. When resuming from a checkpoint, burn-in is skipped automatically since the walkers are already equilibrated.

## See Also

- **Configuration:** [Molecule](#train-sampler), [Solid](#solid-train-sampler), [Hall](#hall-train-sampler)
- **Extending:** <project:/extending/custom-components/samplers.md>
- **API reference:** <project:/api-reference/samplers.md>
