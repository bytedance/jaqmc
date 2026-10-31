# Wavefunction overlap

{class}`~jaqmc.estimator.overlap.Overlap` computes the squared wavefunction overlap between the current wavefunction $\psi_\theta$ and one or more saved reference states $\psi_k$:

$$
\mathcal{O}_k = \frac{|\langle \psi_\theta | \psi_k \rangle|^2}
                     {\langle \psi_\theta | \psi_\theta \rangle
                      \langle \psi_k | \psi_k \rangle}
$$

The overlap satisfies $0 \le \mathcal{O}_k \le 1$: values near $1$ indicate that $\psi_\theta$ and $\psi_k$ represent the same physical state up to normalization and global phase, whereas $\mathcal{O}_k = 0$ indicates orthogonal states. During VMC training, `Overlap` also acts as an orthogonalization penalty that drives $\psi_\theta$ orthogonal to lower-energy reference states so the optimizer converges to excited states.

## Two-distribution estimator

Neural network wavefunctions are unnormalized, so $\mathcal{O}_k$ cannot be evaluated from a single walker distribution alone. `Overlap` rewrites $\mathcal{O}_k$ as a product of two Monte Carlo expectations over the current walkers $x \sim |\psi_\theta|^2$ and an independent reference walker population $y \sim |\psi_k|^2$:

$$
\mathcal{O}_k
  = \frac{\langle \psi_k | \psi_\theta \rangle}{\langle \psi_k | \psi_k \rangle}
    \frac{\langle \psi_\theta | \psi_k \rangle}{\langle \psi_\theta | \psi_\theta \rangle}
  = \operatorname{Re}\!\left(
      \langle Q_k(y) \rangle_{y \sim |\psi_k|^2} \,
      \langle R_k(x) \rangle_{x \sim |\psi_\theta|^2}
    \right)
$$

where the local wavefunction ratios are

$$
R_k(x) = \frac{\psi_k(x)}{\psi_\theta(x)},
\qquad
Q_k(y) = \frac{\psi_\theta(y)}{\psi_k(y)}.
$$

When a reference state is restored from its checkpoint, `Overlap` initializes its reference walker population and sampler state from that checkpoint and advances $y \sim |\psi_k|^2$ on each step using the workflow's configured sampler (`sampler.*`) while keeping the reference parameters fixed. Each reference checkpoint must therefore have a sampler state compatible with the current workflow's `sampler.*` configuration. Both ratios are evaluated in log space and shifted by their batch-wise maximum log difference so arbitrary normalization offsets cancel without overflow or underflow.

The estimator reports each per-state overlap as `overlap:<name>` and the sum over active reference states as `overlap`.

## Orthogonalization penalty

During VMC training, the overlap penalty adds a weighted sum over active reference states to the total loss:

$$
\mathcal{L}_\text{overlap}(\theta)
  = \lambda \sum_k w_k \, \mathcal{O}_k(\theta)
$$

where $\lambda$ is `estimators.overlap.loss_weight` (default `1.0` in training, and omitted in evaluation via {func}`~jaqmc.estimator.HideLoss`) and $w_k$ is `estimators.overlap.states.<k>.weight` (default `1.0`). Setting `loss_weight=0.0` (or `states.<k>.weight=0.0`) tracks the overlap without adding a penalty gradient.

By the symmetry of $\mathcal{O}_k$ between $\langle \psi_\theta | \psi_k \rangle$ and $\langle \psi_k | \psi_\theta \rangle$, differentiating $\mathcal{O}_k(\theta)$ with respect to $\theta$ is equivalent to differentiating the expectation value of the fixed projection operator $\hat{P}_k = \frac{|\psi_k\rangle\langle\psi_k|}{\langle\psi_k|\psi_k\rangle}$, whose local value on current walkers $x \sim |\psi_\theta|^2$ is

$$
P_{k,L}(x) = \langle Q_k(y) \rangle_{y \sim |\psi_k|^2} \, R_k(x).
$$

Each reference state passes $\lambda w_k P_{k,L}(x)$ to {func}`~jaqmc.estimator.loss_grad.loss_term` (<project:loss-grad.md>) for VMC covariance gradient estimation.

## Excited-state workflow

Excited-state VMC runs sequentially. First train the ground state (or lower-lying states), then start a new training run with `estimators.overlap.enabled=true` and point `estimators.overlap.states` to each saved checkpoint directory or `.npz` file:

```bash
# 1. Train the ground state
jaqmc molecule train workflow.save_path=runs/s0

# 2. Train the first excited state orthogonal to s0
jaqmc molecule train \
  workflow.save_path=runs/s1 \
  estimators.overlap.enabled=true \
  estimators.overlap.loss_weight=5.0 \
  estimators.overlap.states.s0.path=runs/s0
```

The effective penalty weight $\lambda w_k$ must exceed the energy gap $E_\text{excited} - E_k$ so that collapsing onto a lower state $\psi_k$ increases the penalized objective $E + \lambda \sum_k w_k \mathcal{O}_k$. Setting $\lambda$ orders of magnitude larger than the gap amplifies gradient variance relative to the energy gradient.

Higher excited states are targeted by adding entries under `estimators.overlap.states`. Each entry is a {class}`~jaqmc.estimator.overlap.ReferenceState` with its own `path`, optional relative `weight` (default `1.0`), and optional `wf` override (when a reference checkpoint was trained with a different wavefunction architecture or hyperparameters than the current workflow):

```yaml
estimators:
  overlap:
    enabled: true
    loss_weight: 5.0
    states:
      s0:
        path: runs/s0
      s1:
        path: runs/s1/train_ckpt_010000.npz
        weight: 1.5
        wf:
          module: lapnet
          ndets: 8
```

In `jaqmc <app> evaluate`, the same `estimators.overlap.enabled=true` and `estimators.overlap.states.<k>.path=...` settings measure overlaps against reference checkpoints without computing penalty gradients.

## See also

- Gradient estimation and outlier clipping: <project:loss-grad.md>
- Training configuration: [Molecule](#train-estimators), [Solid](#solid-train-estimators), [Hall](#hall-train-estimators), [Moire](#moire-train-estimators), [Electron gas](#electron-gas-config)
- Evaluation configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators), [Hall](#hall-estimators), [Moire](#moire-estimators), [Electron gas](#electron-gas-config)
- API: {class}`~jaqmc.estimator.overlap.Overlap`, {class}`~jaqmc.estimator.overlap.ReferenceState`
