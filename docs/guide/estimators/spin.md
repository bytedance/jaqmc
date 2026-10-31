# Spin-squared

The spin-squared estimator ({class}`~jaqmc.estimator.spin.SpinSquared`, reported as `spin_squared`) computes $\langle S^2 \rangle$, the expectation value of the total spin operator. For a proper spin eigenstate with quantum number $S$, this equals $S(S+1)$. Deviations indicate the wavefunction is mixing different spin states, such as a singlet ($S=0$) wavefunction picking up triplet ($S=1$) character ($\langle S^2 \rangle > 0$). This is most relevant for open-shell systems where spin contamination is a known concern.

## Derivation

Starting from the angular-momentum identity:

$$
S^2 = S_z(S_z + 1) + S_- S_+
$$

we expand $S_- S_+ = \sum_{k,l} s_k^- s_l^+$ in first quantization, where $s_l^+$ raises electron $l$ from $\downarrow$ to $\uparrow$ and $s_k^-$ lowers electron $k$ from $\uparrow$ to $\downarrow$.

**Diagonal terms** ($k = l$): $s_l^- s_l^+$ is the identity on each $\downarrow$ electron, contributing $+n_\downarrow$ total.

**Off-diagonal terms** ($k \ne l$, with $k \in \uparrow$ and $l \in \downarrow$): applying $s_l^+$ then $s_k^-$ swaps the spin labels of electrons $k$ and $l$. For an antisymmetric wavefunction whose first $n_\uparrow$ coordinates are spin-up, this is equivalent to swapping the spatial coordinates of electrons $k$ and $l$ up to a minus sign from antisymmetry, so each swap ratio $\Psi(\mathbf{r}_{k\leftrightarrow l})/\Psi(\mathbf{r})$ enters with a negative sign relative to the $+n_\downarrow$ diagonal terms.

Combining gives the local estimator:

$$
S^2_\text{local}
    = S_z(S_z + 1) + n_\downarrow
      - \sum_{k \in \uparrow} \sum_{l \in \downarrow}
        \frac{\Psi(\mathbf{r}_{k \leftrightarrow l})}
             {\Psi(\mathbf{r})}
$$

## Computational details

**Minority-channel optimization.** The double sum is symmetric in the swap, so the outer loop runs over the minority spin channel (with $n_\text{min}$ electrons) to minimize wavefunction evaluations. Using $|S_z|$ in the prefactor compensates:

$$
|S_z|(|S_z|+1) + n_\text{min} = S_z(S_z+1) + n_\downarrow
$$

**Log-space ratios.** The network represents $\Psi = \sigma \, e^{\log|\Psi|}$ where $\sigma$ is the phase ($\pm 1$ for real wavefunctions, $e^{i\phi}$ for complex). Writing $\Psi'$ for the swapped configuration:

$$
\frac{\Psi'}{\Psi}
  = \frac{\sigma'}{\sigma}
    \cdot \exp\!\bigl(\log|\Psi'| - \log|\Psi|\bigr)
$$

The sum over swaps shifts the exponents by $\max(\log|\Psi'| - \log|\Psi|)$ for numerical stability.

## Single-electron mode

In `"full"` mode (the default), evaluating `SpinSquared` requires $n_\text{min} \times n_\text{maj}$ wavefunction calls per walker. Setting `mode="single_electron"` reduces this cost to $n_\text{maj}$ calls per walker by evaluating one minority electron per step ($i = \text{step} \bmod n_\text{min}$) and scaling its swap sum by $n_\text{min}$:

$$
S^{2,(i)}_\text{local}
    = |S_z|(|S_z| + 1)
      + n_\text{min} \left(
          1 - \sum_{j \in \text{maj}}
              \frac{\Psi(\mathbf{r}_{i \leftrightarrow j})}
                   {\Psi(\mathbf{r})}
        \right)
$$

Averaging over $n_\text{min}$ steps recovers $\langle S^2 \rangle$.

## Spin penalty during training

During VMC training, {class}`~jaqmc.estimator.spin.SpinSquared` can penalize spin contamination by adding $w \langle P_L \rangle^2$ (where $P_L = S^2_\text{local} - |S_z|(|S_z|+1)$ is the non-constant $S_- S_+$ contribution) through `loss_weight` ({func}`~jaqmc.estimator.loss_grad.loss_term`), with parameter gradient $2 w \langle P_L \rangle \nabla_\theta \langle P_L \rangle$. Because $P_L$ depends on wavefunction swap ratios, $\nabla_\theta \langle P_L \rangle$ includes both the standard VMC covariance term and the direct derivative $\langle \nabla_\theta P_L \rangle$. Enabling the estimator (`estimators.spin.enabled=true`) and setting a nonzero `loss_weight` (default `0.0`, for example `estimators.spin.loss_weight=1.0`) activates the spin penalty.

## See also

- Gradient estimation and outlier clipping: <project:loss-grad.md>
- Training configuration: [Molecule](#train-estimators), [Solid](#solid-train-estimators), [Electron gas](#electron-gas-config)
- Evaluation configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators), [Electron gas](#electron-gas-config)
- API: {class}`~jaqmc.estimator.spin.SpinSquared`
