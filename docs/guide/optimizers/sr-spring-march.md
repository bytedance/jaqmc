# Robust SR, SPRING, and MARCH

This page explains the deeper logic behind JaQMC's stochastic reconfiguration (SR) optimizer. Read it when the high-level guidance in {doc}`index` is not enough and you want to understand why robust SR is more stable than plain SR, how SPRING modifies the update, or what MARCH changes.

## What Problem Robust SR Solves

Standard SR preconditions the raw gradient with the inverse Fisher matrix:

$$
\delta_{\mathrm{SR}} = (O^T O + \lambda I)^{-1}\tilde{\delta}
$$

Here:

- $O$ is the centered score matrix built from log-wavefunction gradients, scaled by $1 / \sqrt{N_{\mathrm{batch}}}$.
- $\tilde{\delta}$ is the raw gradient passed to the optimizer.
- $\lambda$ is the damping term.

This works well when the gradient lies mostly in the span of the score matrix. The problem is the null space: if a gradient component does not align with the wavefunction tangent space, the inverse can amplify it by roughly $1 / \lambda$. With the small damping values common in practice, that can turn sampling noise or non-Hermitian effects into an unstable update.

## Robust SR: Natural Gradient Where Curvature Is Trustworthy

JaQMC uses a robust SR formulation that blends natural gradient and standard gradient descent instead of applying the same inverse everywhere.

Let

$$
F = O O^T + \lambda I.
$$

The robust SR operator is

$$
\delta_k = \left(I - O^T F^{-1} O + O^T F^{-2} O\right)\tilde{\delta}.
$$

The key idea is a soft switch:

- In high-curvature directions, the update behaves like standard SR.
- In weak or noisy directions, it falls back toward ordinary gradient descent instead of exploding.

That behavior is visible in the effective eigenvalue scaling

$$
G(h) = 1 - \frac{h}{h + \lambda} + \frac{h}{(h + \lambda)^2},
$$

where $h$ is the curvature along one mode:

- If $h \gg \lambda$, then $G(h) \approx 1 / h$, which recovers SR-like natural-gradient scaling.
- If $h \ll \lambda$, then $G(h) \approx 1$, which falls back to SGD-like scaling.

## SPRING: Add Momentum Without Overshooting Stiff Directions

SPRING adds a momentum term

$$
p = \mu \delta_{k-1}
$$

but does not treat that momentum as a free Euclidean update. Instead, the momentum is projected through the same tangent-space geometry so that it does not keep pushing strongly in directions where the current curvature already provides a better correction.

In operator form:

$$
\delta_k = \mathbf{G}\tilde{\delta} + \left(I - O^T F^{-1} O\right)p,
$$

where $\mathbf{G}$ is the robust SR operator above.

In practice, that means:

- flat directions keep useful momentum,
- stiff directions get less momentum carry-over,
- the optimizer is less likely to oscillate when curvature is large.

## MARCH: Add an Adaptive Per-Parameter Metric

MARCH introduces a diagonal metric $M$ so that volatile parameters are scaled down before the SR solve.

JaQMC supports two ways to estimate that metric:

- `march_mode=diff`: uses squared update differences, $(\delta_k - \delta_{k-1})^2$.
- `march_mode=var`: uses score variance along the sample axis, derived from the diagonal of $O^T O$.

The optimizer stores an exponential moving average

$$
v_k = \beta v_{k-1} + (1 - \beta)\tilde{v},
$$

then forms the diagonal scaling

$$
M^{-1} = \operatorname{diag}\left(\frac{1}{\sqrt{v_k + \epsilon}}\right).
$$

The result is a hybrid update:

- strong modes still get natural-gradient behavior,
- weak or noisy modes get a more conservative adaptive first-order step,
- parameters with unstable history are automatically damped more strongly.

This is why MARCH often feels like "SR where the metric adapts to parameter volatility" rather than a completely different optimizer.

## The Efficient Kernel Form JaQMC Actually Uses

JaQMC does not form large dense parameter-space operators directly. Internally, it rewrites the update in kernel form and solves a batch-sized linear system instead.

The core update uses:

$$
u_g = O(M^{-1}\tilde{\delta}), \qquad
u_p = O p, \qquad
F_M = O M^{-1} O^T + \lambda I.
$$

Then JaQMC solves

$$
w = F_M^{-1}u_g, \qquad
z = F_M^{-1}(u_g + u_p - w),
$$

and finishes with

$$
\delta_k = M^{-1}\tilde{\delta} + p - M^{-1}O^T z.
$$

That is the same structure JaQMC uses internally.

Why this matters:

- the expensive work is reduced to score-matrix products and a batch-sized Cholesky solve,
- memory can be controlled with `score_chunk_size` and `gram_num_chunks`,
- the code can adapt damping with `max_cond_num` before solving the system.

## Tuning Guide

Start with JaQMC's SR defaults. They already enable robust SR together with SPRING and MARCH, which is usually the right production baseline.

Reach for these settings when something specific goes wrong:

- Training is unstable or the SR solve looks ill-conditioned: increase `damping` or tighten `max_cond_num`.
- Training is stable but too sluggish: try reducing `damping` slightly, or compare against KFAC before spending too much time hand-tuning SR.
- Momentum seems to cause overshoot or oscillation: lower `spring_mu` or disable SPRING with `train.optim.spring_mu=null`.
- You want simpler SR behavior for an ablation or comparison: disable MARCH with `train.optim.march_beta=null`.
- Memory use is too high: reduce `score_chunk_size` or increase `gram_num_chunks`.

`march_mode=var` is the default because it builds the adaptive metric from current score variance, which is usually a good general-purpose choice. Use `march_mode=diff` mainly when you specifically want the metric to track update volatility instead.
