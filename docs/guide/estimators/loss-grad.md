# Loss and gradient

During VMC training, gradient-enabled estimators ({class}`~jaqmc.estimator.total_energy.TotalEnergy`, {class}`~jaqmc.estimator.overlap.Overlap`, {class}`~jaqmc.estimator.spin.SpinSquared`, and {class}`~jaqmc.estimator.angular_momentum.SphericalAngularMomentum`) compute parameter gradients for VMC optimization using the shared helpers in {mod}`jaqmc.estimator.loss_grad`. For energy minimization, the goal is to find wavefunction parameters $\theta$ that minimize the variational energy

$$
E(\theta) = \frac{\langle \psi_\theta | H | \psi_\theta \rangle}
                  {\langle \psi_\theta | \psi_\theta \rangle}
$$

In VMC, $E(\theta)$ is estimated by averaging the local energy $E_L = H\psi/\psi$ over walker positions sampled from $|\psi_\theta|^2$.

To minimize $E(\theta)$, we need its gradient with respect to $\theta$. Because the sampling distribution $|\psi_\theta|^2$ depends on $\theta$, we cannot simply backpropagate through the local energy average. Accounting for the sampling distribution, the gradient of the variational energy is:

$$
\nabla_\theta E
  = 2 \operatorname{Re} \left\langle
      (\nabla_\theta \log\psi_\theta)^* \left(E_L - \left\langle E_L \right\rangle\right)
  \right\rangle
$$

where $\langle \cdot \rangle$ is the Monte Carlo average over walkers. Each walker contributes its log-wavefunction gradient $\nabla_\theta \log\psi_\theta$, weighted by how far its local energy deviates from the mean. For real wavefunctions this reduces to $2 \langle (E_L - \langle E_L \rangle) \nabla_\theta \log|\psi_\theta| \rangle$.

The $\langle E_L \rangle$ baseline arises from differentiating the ratio $\langle \psi|H|\psi \rangle / \langle \psi|\psi \rangle$ (quotient rule) and accounts for the fact that neural network wavefunctions are not normalized. It also reduces the variance of the gradient estimate, since walkers near the mean energy contribute little.

Penalty estimators contribute weighted local terms via {func}`~jaqmc.estimator.loss_grad.loss_term` (`loss_weight` on each estimator, plus the direct derivative $\langle \nabla_\theta \mathcal{O}_L \rangle$ when $\mathcal{O}_L$ itself depends on $\theta$, and any nonlinear penalty wrapper $f(\langle \mathcal{O}_L \rangle)$). During training, {class}`~jaqmc.estimator.loss_grad.LossGrad` sums all active $\partial \mathcal{L} / \partial \log\psi$ terms into a single effective local Hamiltonian while keeping `total_energy` as the unpenalized variational energy.

By default, {class}`~jaqmc.estimator.loss_grad.LossGrad` evaluates $\log\psi$ and $\nabla_\theta \log\psi$ at the wavefunction's default precision (`wf.precision`). Set `train.grads.wavefunction_precision` (default `null`) to override the mode for gradient evaluation (see <project:../mixed-precision.md>).

## Outlier clipping

Before centering the effective local Hamiltonian $H_{\text{eff},L}$ for the covariance gradient, {class}`~jaqmc.estimator.loss_grad.LossGrad` (`train.grads.*`) clips $H_{\text{eff},L}$ across the walker batch to suppress extreme outliers. This affects the gradient only, while reported losses remain unclipped averages. The following methods can be chosen via `train.grads.clip_method`:

- `mad` (Default): clips to $[\mathrm{median}(H_{\text{eff},L}) - s \cdot \mathrm{median}(|H_{\text{eff},L} - \mathrm{median}(H_{\text{eff},L})|),\; \mathrm{median}(H_{\text{eff},L}) + s \cdot \mathrm{median}(|H_{\text{eff},L} - \mathrm{median}(H_{\text{eff},L})|)]$.
- `iqr`: clips to $[Q_1 - s \cdot \mathrm{IQR},\; Q_3 + s \cdot \mathrm{IQR}]$, where $Q_1$ and $Q_3$ are the quartiles and `s` is `train.grads.clip_scale` (default `5.0`).
- `none`: disables clipping entirely.

Decreasing `clip_scale` clips more aggressively, which stabilises gradients but biases the estimator.

## See also

- Penalty estimators: <project:overlap.md>, <project:spin.md>, [Angular momentum penalties](../../systems/hall/index.md#angular-momentum-penalties)
- Configuration: [Molecule](#train-grads), [Solid](#solid-train-grads), [Hall](../../systems/hall/train.md), [Moire](../../systems/moire/train.md), [Electron gas](#electron-gas-config)
- API: {func}`~jaqmc.estimator.loss_grad.loss_term`, {class}`~jaqmc.estimator.loss_grad.LossGrad`, {class}`~jaqmc.estimator.total_energy.TotalEnergy`
