# Optimizers

The optimizer updates the wavefunction parameters to minimize the variational energy. Choosing the right optimizer significantly affects convergence speed and stability.

## Natural Gradient vs Standard Gradient

Standard optimizers like Adam or SGD compute gradients in the space of model parameters (weights and biases). A small change in parameter space can produce a large, uneven change in the wavefunction: stable in some regions, unstable in others.

**Natural gradient** methods (SR, KFAC) instead compute updates in the space of the wavefunction itself, using the Fisher information matrix to account for how parameter changes affect the physical state. This produces more uniform, stable updates and typically converges faster for VMC.

The tradeoff is cost: natural gradient methods are more expensive per step. For small test runs, Adam is faster to iterate. For production simulations, KFAC or SR usually reach lower energies in fewer steps.

| | KFAC | SR (Robust SR + SPRING/MARCH) | Adam / LAMB |
|---|------|--------------------------------|-------------|
| **Convergence** | Fast | Fast | Slower |
| **Cost per step** | Moderate | Moderate | Low |
| **Stability** | High | High | Can be unstable for stiff systems |
| **Recommended for** | Production runs | Production runs, exact natural gradient | Pretraining, quick tests |

**Rule of thumb:** Start with KFAC (the default). SR with SPRING/MARCH is a strong alternative. Try both and compare convergence for your system. Use Optax optimizers for pretraining or quick experiments where per-step cost matters more than convergence rate.

## Available Optimizers

### KFAC

[Kronecker-Factored Approximate Curvature](https://arxiv.org/abs/1503.05671) approximates the Fisher information matrix as a Kronecker product of smaller per-layer matrices, making natural gradient tractable for deep networks. For complex-valued wavefunctions (for example solids with Bloch phases), JaQMC patches the curvature blocks to better match the wavefunction geometry rather than using the standard FIM directly. This happens automatically.

**When to use:** Production runs. KFAC is the most battle-tested optimizer for neural-network VMC.

KFAC is the default optimizer, so no configuration change is needed. The key tunable parameters are:

- `damping` (default `1e-3`) - Tikhonov regularization added to the curvature factors before inversion. Larger values improve conditioning when local energies fluctuate strongly.
- `norm_constraint` (default `1e-3`) - cap on the approximate squared Fisher norm of each parameter step.
- `inverse_update_period` (default `1`) - number of training steps between Kronecker-factor inverse updates. Curvature moving averages still update every step while cached factor inverses are reused in between.
- `use_cholesky_inversion` (default `false`) - whether damped Kronecker factors are inverted with a positive-definite Cholesky solver instead of general LU inversion.
- `wavefunction_precision` (default `null`) - wavefunction precision mode used when evaluating curvature (`null` inherits `wf.precision`, see <project:../mixed-precision.md>).

For large wavefunctions where KFAC factor inversion takes a noticeable share of step time, Cholesky inversion and a larger inverse update period reduce per-step overhead on GPUs:

```bash
train.optim.use_cholesky_inversion=true train.optim.inverse_update_period=2
```

Values of `inverse_update_period` around `2` to `5` amortize factor inversion across steps with little effect on steady-state convergence, whereas much larger periods can leave cached inverses stale when the wavefunction changes rapidly. Enabling `use_cholesky_inversion=true` skips `kfac-jax`'s per-factor `NaN`/`Inf` conditional check and relies on the damped factors remaining strictly positive-definite. The default `damping=1e-3` typically keeps factors well-conditioned, whereas very small damping values (below `1e-4`) or severe local-energy spikes can cause non-positive-definite factors and `NaN` updates.

Built-in `wavefunction_precision` modes work with KFAC out of the box because master parameters remain `float32`. For custom wavefunctions or `jax.enable_x64` runs, JaQMC's KFAC tags cover parameters and inputs that share the same dtype, as well as `float32` parameters receiving `float64` inputs or producing `float64` values. Other custom parameter/input dtype combinations may need additional validation against the `float32` graph registration.

See the [configuration reference](#train-optim) for all KFAC options.

### Stochastic Reconfiguration (SR)

[Stochastic reconfiguration](https://doi.org/10.1103/PhysRevLett.80.4558) updates parameters following imaginary-time evolution. JaQMC's SR implementation supports a configurable robust-SR fallback together with the [SPRING](https://doi.org/10.1016/j.jcp.2024.113351) momentum extension and the [MARCH](http://arxiv.org/abs/2507.02644) adaptive metric.

By default, SR uses `robust_gamma=sqrt`, `spring_mu=0.9`, `march_beta=0.995`, `march_mode=var`, `max_norm=fixed`, and adaptive damping bounded by `max_cond_num=1e7`. To disable SPRING or MARCH, set the corresponding field to `null`:

```bash
train.optim.spring_mu=null
train.optim.march_beta=null
```

`max_norm=fixed` means the optimizer normalizes each step to the norm implied by `learning_rate`. Set `train.optim.max_norm=<float>` if you want an explicit norm cap instead, or `train.optim.max_norm=null` to remove the norm constraint entirely.

**When to use:** When you want an exact SR-style natural-gradient update instead of KFAC's structured approximation.

To use SR instead of the default KFAC:

```bash
# Default FP64 Gram construction and solve
jaqmc <app> train train.optim.module=sr

# Fast mixed-precision SR optimizer step (add train.optim.gram_solve_prec=F32_PCG on non-FP64 GPUs such as H20)
jaqmc <app> train train.optim.module=sr \
  train.optim.gram_dot_prec=F16 \
  train.optim.wavefunction_precision=fp16_mixed
```

To accelerate the entire VMC step (including kinetic energy and MCMC sampling alongside SR), see <project:../mixed-precision.md>.

The key tunable parameters are:

- `damping` (default `1e-3`) - regularization for the Gram matrix.
- `robust_gamma` (default `"sqrt"`) - controls how strongly low-curvature directions fall back from SR toward first-order scaling. Set `null` for standard damped SR or `1.0` for the more conservative legacy robust-SR behavior.
- `max_norm` (default `"fixed"`) - ties the update norm to `learning_rate`. Set a float for an explicit cap or `null` to disable the norm constraint.
- `max_cond_num` (default `1e7`) - adaptively raises damping to limit Gram-matrix conditioning.
- `score_chunk_size` (default `null`) / `gram_num_chunks` (default `1`) - trade throughput for lower peak memory on large systems. Keep `score_chunk_size` at `null` (or as large a walker chunk as fits in memory, such as `128`–`512`) and `gram_num_chunks` small (`1`–`4`) to avoid extra backward-pass or multi-GPU `all_to_all` overhead.
- `gram_dot_prec` (default `"F64"`) - precision mode for Gram matrix inner products. Set `"F16"` (or `"F16_X3"` under very small damping, $\lambda \le 10^{-5}$) to accelerate the Gram matrix product on FP16 Tensor Cores (see [Gram Matrix Precision](#sr-gram-precision)).
- `gram_solve_prec` (default `"F64"`) - precision mode for Gram matrix accumulation and inversion (`"F64"`, `"F32_PCG"`, or `"F32"`). Keep `"F64"` on GPUs with full-rate FP64 Tensor Cores (such as A100 or H100), and set `"F32_PCG"` on GPUs with limited FP64 throughput (such as H20, see [Gram Matrix Precision](#sr-gram-precision)).
- `wavefunction_precision` (default `null`) - wavefunction precision mode used for score evaluation (`null` inherits `wf.precision`, or set `"fp16_mixed"` to accelerate `grad_logpsi`, see <project:../mixed-precision.md>).

For the derivation, failure mode, and efficient kernel-form update used by JaQMC, see {doc}`sr-spring-march`.

### Optax Optimizers

JaQMC supports any optimizer from [Optax](inv:optax:*:doc#index) - Adam, LAMB, SGD, AdamW, and others. These are standard first-order optimizers.

**When to use:** Quick experiments, pretraining, or when natural-gradient methods are too expensive for your system size.

To use an Optax optimizer, set the module to `optax:<name>`:

```bash
train.optim.module=optax:adam
```

## Learning Rate Schedules

All optimizers support swappable learning-rate schedules via `train.optim.learning_rate.module`.

**`schedule:Standard`** (default) - inverse-power decay:

$$\eta_t = \texttt{rate} \cdot \left(\frac{1}{1 + t \,/\, \texttt{delay}}\right)^{\texttt{decay}}$$

Defaults: `rate` = 0.05, `delay` = 2000, `decay` = 1. Override with `train.optim.learning_rate.<param>`:

```bash
train.optim.learning_rate.rate=0.01 train.optim.learning_rate.delay=5000
```

**`schedule:Constant`** - fixed learning rate throughout training. Override with `train.optim.learning_rate.rate` (default 0.05).

```{note}
The optimizer controls how gradients are *applied*. Gradient *estimation*, including loss clipping via `train.grads.clip_method` and `train.grads.clip_scale`, is handled separately. See <project:../estimators/loss-grad.md> for how gradients are computed and stabilised before the optimizer sees them.
```

## See Also

- **Configuration:** {ref}`Molecule <train-optim>`, {ref}`Solid <solid-train-optim>`, {ref}`Hall <hall-train-optim>`
- **Extending:** <project:/extending/custom-components/optimizers.md>
- **API reference:** <project:/api-reference/optimizers.md>

```{toctree}
:hidden:
:maxdepth: 1

sr-spring-march.md
```
