# Robust SR, SPRING, and MARCH

This page explains the deeper logic behind JaQMC's stochastic reconfiguration (SR) optimizer. Read it when the high-level guidance in {doc}`index` is not enough and you want to understand why robust SR is more stable than plain SR, how `robust_gamma` changes that fallback, how SPRING modifies the update, or what MARCH changes.

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

JaQMC uses a robust SR formulation that blends natural gradient and first-order fallback behavior instead of applying the same inverse everywhere. The blend is controlled by `robust_gamma`.

Let

$$
F = O O^T + \lambda I.
$$

The robust SR operator is

$$
\delta_k =
\left(
\gamma I
- \gamma O^T F^{-1} O
+ (1 - \gamma \lambda) O^T F^{-2} O
\right)\tilde{\delta}.
$$

Here `robust_gamma` chooses the low-curvature fallback:

- `robust_gamma=null` recovers standard damped SR.
- `robust_gamma=1.0` gives the older robust-SR behavior, which falls back to ordinary gradient descent in weak modes.
- `robust_gamma="sqrt"` is the current default. It falls back more aggressively than gradient descent but less aggressively than standard SR.

The key idea is still a soft switch:

- In high-curvature directions, the update behaves like standard SR.
- In weak or noisy directions, it falls back toward ordinary gradient descent instead of exploding.

That behavior is visible in the effective eigenvalue scaling

$$
G_\gamma(h) = \gamma - \gamma \frac{h}{h + \lambda} + (1 - \gamma \lambda)\frac{h}{(h + \lambda)^2}
= \frac{h + \gamma \lambda^2}{(h + \lambda)^2},
$$

where $h$ is the curvature along one mode:

- If $h \gg \lambda$, then $G_\gamma(h) \approx 1 / h$, which recovers SR-like natural-gradient scaling.
- If $h \ll \lambda$, then $G_\gamma(h) \approx \gamma$, so `robust_gamma` directly sets the weak-mode fallback.

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
z = F_M^{-1}(\gamma u_g + u_p - (1 - \gamma \lambda) w),
$$

and finishes with

$$
\delta_k = \gamma M^{-1}\tilde{\delta} + p - M^{-1}O^T z.
$$

That is the same structure JaQMC uses internally.

Why this matters:

- the expensive work is reduced to score-matrix products and a batch-sized Cholesky solve,
- memory can be controlled with `score_chunk_size` and `gram_num_chunks`,
- the code can adapt damping with `max_cond_num` before solving the system.

(sr-gram-precision)=
### Gram Matrix Precision

Forming the $N_{\mathrm{batch}} \times N_{\mathrm{batch}}$ Gram matrix $O M^{-1} O^T$ ($O(N_{\mathrm{batch}}^2 N_{\mathrm{params}})$) and solving the dual linear system $(O M^{-1} O^T + \lambda I)^{-1}$ ($O(N_{\mathrm{batch}}^3)$) are the two main linear-algebra costs inside an SR update.

JaQMC exposes separate precision controls for these two stages:

```bash
jaqmc molecule train train.optim.module=sr \
  train.optim.gram_dot_prec=F16 \
  train.optim.gram_solve_prec=F32_PCG
```

#### 1. Gram Matrix Inner Products (`gram_dot_prec`)

`gram_dot_prec` selects the precision used to multiply the score matrices within each parameter chunk:

| `gram_dot_prec` | How inner products are computed | Relative Gram error | Update error at $\lambda = 10^{-3}$ / $10^{-5}$ | When to choose |
|---|---|---|---|---|
| `F64` (default) | Double precision (FP64) | Exact reference | Exact reference | Exact reference on GPUs with full-rate FP64 Tensor Cores (such as A100 or H100), or small models where Gram construction is already fast. |
| `F32` | Single precision (FP32) | $\sim 10^{-7}$ | $\sim 10^{-8}$ / $\sim 10^{-6}$ | Conservative non-FP64 baseline. |
| `F16_X3` | 3-pass compensated FP16 (high + low remainder split) | $\sim 10^{-6}$ | $\sim 10^{-8}$ / $\sim 10^{-6}$ | Very small damping ($\lambda \le 10^{-5}$) or stiff systems that need near-FP32 update accuracy ($\sim 2.2\times$ faster than `F64` on A100 and $\sim 1.3\times$ faster than `F32` on H20). |
| `F16` | 1-pass rescaled FP16 with blocked FP32 summation | $\sim 10^{-6}$ | $\sim 10^{-7}$ / $\sim 10^{-5}$ | **Recommended for fast Gram construction** under standard damping ($\lambda \ge 10^{-4}$). Roughly $4.5\times$ faster than `F64` on A100 and $3\times$ to $4\times$ faster than `F32` on H20, and halves cross-device `all_to_all` traffic by communicating score chunks in FP16. |

Both `F16` and `F16_X3` rescale each score row by an exact power of two before casting to FP16 and sum products in 512-parameter blocks in FP32. While both modes reach a similar overall Frobenius Gram error ($\sim 10^{-6}$, dominated by large eigenvalues), the inverse $(O M^{-1} O^T + \lambda I)^{-1}$ amplifies weak-curvature directions by $1 / \lambda$. Along those directions, `F16_X3` splits each entry into a primary FP16 value and an FP16 remainder to preserve cancellation, reducing update error by $\sim 10\times$ at $\lambda \le 10^{-5}$.

#### 2. Gram Accumulation and Linear Solve (`gram_solve_prec`)

`gram_solve_prec` controls how the $N_{\mathrm{batch}} \times N_{\mathrm{batch}}$ Gram matrix is accumulated across chunks and inverted:

| `gram_solve_prec` | Gram accumulation & $(G + \lambda I)$ | $O(N_{\mathrm{batch}}^3)$ factorization | Refinement | When to choose |
|---|---|---|---|---|
| `F64` (default) | FP64 | Direct FP64 Cholesky | None | GPUs with full-rate FP64 compute (A100, H100), or small batch sizes where FP64 Cholesky is already negligible. |
| `F32_PCG` | FP64 | FP32 Cholesky preconditioner | FP64 preconditioned conjugate gradient (PCG) | **Recommended on GPUs with limited FP64 throughput** (such as H20 or workstation GPUs). Runs the expensive $O(N_{\mathrm{batch}}^3)$ factorization in FP32 and recovers full FP64 solver accuracy in a few $O(N_{\mathrm{batch}}^2)$ FP64 PCG steps ($6\times$ to $7.5\times$ faster solve at $N_{\mathrm{batch}} = 4096$). |
| `F32` | FP32 | Direct FP32 Cholesky | None | Pure-FP32 environments when `x64` is disabled and damping is relatively large ($\lambda \ge 10^{-2}$). |

## Tuning Guide

Start with JaQMC's SR defaults. They use `robust_gamma="sqrt"` together with SPRING, MARCH, adaptive damping, and `max_norm="fixed"`, which is usually the right production baseline.

Reach for these settings when something specific goes wrong:

- Training is unstable or the SR solve looks ill-conditioned: increase `damping` or tighten `max_cond_num`.
- Training is stable but too sluggish: try reducing `damping` slightly, or try `train.optim.robust_gamma=null` if you want weaker fallback and behavior closer to standard damped SR.
- Weak-curvature directions look too jumpy: try `train.optim.robust_gamma=1.0` for the more conservative legacy robust-SR fallback.
- Momentum seems to cause overshoot or oscillation: lower `spring_mu` or disable SPRING with `train.optim.spring_mu=null`.
- You want simpler SR behavior for an ablation or comparison: disable MARCH with `train.optim.march_beta=null`.
- You want the step norm tied directly to the learning-rate schedule: keep `train.optim.max_norm=fixed`. Use a float if you want an explicit cap, or `null` if you want no norm constraint.
- Memory use is too high: lower `score_chunk_size` (for example to `128` or `256` walkers per backward chunk) or increase `gram_num_chunks` (for example from `1` to `2` or `4` parameter chunks). Keep chunks as large and few as fits in GPU memory, since tiny `score_chunk_size` values serialize `grad_logpsi` and large `gram_num_chunks` values (`16+`) add multi-GPU `all_to_all` overhead.
- SR step time is high: set `train.optim.gram_dot_prec=F16` (or `F16_X3` when $\lambda \le 10^{-5}$) and `train.optim.wavefunction_precision=fp16_mixed`. On GPUs without full-rate FP64 Tensor Cores (such as H20), also set `train.optim.gram_solve_prec=F32_PCG`.

`march_mode=var` is the default because it builds the adaptive metric from current score variance, which is usually a good general-purpose choice. Use `march_mode=diff` mainly when you specifically want the metric to track update volatility instead.
