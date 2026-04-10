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

- `damping` (default `1e-3`) - regularization added to the curvature matrix.
- `norm_constraint` (default `1e-3`) - scales the update down so that its approximate squared Fisher norm is at most this value.

See the [configuration reference](#train-optim) for all KFAC options.

### Stochastic Reconfiguration (SR)

[Stochastic reconfiguration](https://doi.org/10.1103/PhysRevLett.80.4558) updates parameters following imaginary-time evolution. JaQMC's SR implementation supports a robust SR formulation together with the [SPRING](https://doi.org/10.1016/j.jcp.2024.113351) momentum extension and the [MARCH](http://arxiv.org/abs/2507.02644) adaptive metric.

By default, SR enables `spring_mu=0.9`, `march_beta=0.5`, `march_mode=var`, and adaptive damping bounded by `max_cond_num=1e7`. To disable SPRING or MARCH, set the corresponding field to `null`:

```bash
train.optim.spring_mu=null
train.optim.march_beta=null
```

**When to use:** When you want an exact SR-style natural-gradient update instead of KFAC's structured approximation.

To use SR instead of the default KFAC:

```bash
train.optim.module=sr
```

The key tunable parameters are:

- `damping` (default `1e-3`) - regularization for the Gram matrix.
- `max_norm` (default `0.1`) - constrains the update norm.
- `max_cond_num` (default `1e7`) - adaptively raises damping to limit Gram-matrix conditioning.
- `score_chunk_size` / `gram_num_chunks` - trade speed for lower memory use on large systems.

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
The optimizer controls how gradients are *applied*. Gradient *estimation*, including outlier clipping via `clip_scale`, is handled separately. See <project:../estimators/loss-grad.md> for how gradients are computed and stabilised before the optimizer sees them.
```

## See Also

- **Configuration:** [Molecule](#train-optim), [Solid](#solid-train-optim), [Hall](#hall-train-optim)
- **Extending:** <project:/extending/custom-components/optimizers.md>
- **API reference:** <project:/api-reference/optimizers.md>

```{toctree}
:hidden:
:maxdepth: 1

sr-spring-march.md
```
