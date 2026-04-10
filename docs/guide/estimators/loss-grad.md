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

Before computing the gradient, local energies are clipped using the interquartile range (IQR) to suppress extreme outliers. Any local energy falling outside $[Q_1 - s \cdot \mathrm{IQR},\; Q_3 + s \cdot \mathrm{IQR}]$ is clamped to the nearest boundary, where $Q_1$ and $Q_3$ are the first and third quartiles and $s$ is the `clip_scale` parameter.

The default `clip_scale` of 5 is permissive enough to leave most walkers untouched while preventing rare walkers in low-probability regions from dominating the gradient. Decreasing it clips more aggressively, which stabilises gradients but biases the energy estimate. Setting a very large value (e.g. `1e8`) effectively disables clipping.

## See also

- Configuration: [Molecule](#train-grads), [Solid](#solid-train-grads)
- API: {class}`~jaqmc.estimator.loss_grad.LossAndGrad`
