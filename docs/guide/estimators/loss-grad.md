# Loss and gradient

The {class}`~jaqmc.estimator.loss_grad.LossAndGrad` estimator computes parameter gradients for VMC optimization. The goal is to find wavefunction parameters $\theta$ that minimize the variational energy

$$
E(\theta) = \frac{\langle \psi_\theta | H | \psi_\theta \rangle}
                  {\langle \psi_\theta | \psi_\theta \rangle}
$$

In VMC, $E(\theta)$ is estimated by averaging the local energy $E_L = H\psi/\psi$ over walker positions sampled from $|\psi_\theta|^2$.

To minimize $E(\theta)$, we need its gradient with respect to $\theta$. Because the sampling distribution $|\psi_\theta|^2$ depends on $\theta$, we cannot simply backpropagate through the local energy average — changing parameters also changes which configurations the walkers visit. Accounting for this, the gradient of the variational energy is:

$$
\nabla_\theta E
  = 2 \left\langle
      \left(E_L - \left\langle E_L \right\rangle\right) \, \nabla_\theta \log|\psi_\theta|
  \right\rangle
$$

where $\langle \cdot \rangle$ is the Monte Carlo average over walkers. Each walker contributes its log-wavefunction gradient $\nabla_\theta \log|\psi_\theta|$, weighted by how far its local energy deviates from the mean.

The $\langle E_L \rangle$ baseline arises from differentiating the ratio $\langle \psi|H|\psi \rangle / \langle \psi|\psi \rangle$ (quotient rule) — it accounts for the fact that neural network wavefunctions are not normalized. It also reduces the variance of the gradient estimate, since walkers near the mean energy contribute little.

## Outlier clipping

Before computing the gradient, `LossAndGrad` can clip local energies to suppress extreme outliers. This affects the gradient only; the reported loss remains the unclipped energy average. The following methods can be chosen via `clip_method`:

- `mad` (Default): clips to $[\mathrm{median}(E_L) - s \cdot \mathrm{median}(|E_L - \mathrm{median}(E_L)|),\; \mathrm{median}(E_L) + s \cdot \mathrm{median}(|E_L - \mathrm{median}(E_L)|)]$.
- `iqr`: clips to $[Q_1 - s \cdot \mathrm{IQR},\; Q_3 + s \cdot \mathrm{IQR}]$, where $Q_1$ and $Q_3$ are the quartiles and `s` is `clip_scale`.
- `none`: disables clipping entirely.

It's also possible to decrease `clip_scale` to clip more aggressively, which stabilises gradients but biases the estimator. Setting `clip_method="none"` disables clipping explicitly.

## See also

- Configuration: [Molecule](#train-grads), [Solid](#solid-train-grads), [Hall](../../systems/hall/train.md)
- API: {class}`~jaqmc.estimator.loss_grad.LossAndGrad`
