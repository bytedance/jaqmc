# Spin-squared

For configuration options, see the estimator sections of each system's config reference ([molecule](#molecule-estimators), [solid](#solid-estimators)).

The spin-squared estimator computes $\langle S^2 \rangle$, the expectation value of the total spin operator. For a proper spin eigenstate with quantum number $S$, this equals $S(S+1)$. Deviations indicate the wavefunction is mixing different spin states — for example, a singlet ($S=0$) wavefunction that picks up triplet ($S=1$) character would show $\langle S^2 \rangle > 0$. This is most relevant for open-shell systems where spin contamination is a known concern.

## Derivation

Starting from the angular-momentum identity:

$$
S^2 = S_z(S_z + 1) + S_- S_+
$$

we expand $S_- S_+ = \sum_{k,l} s_k^- s_l^+$ in first quantization, where $s_l^+$ raises electron $l$ from $\downarrow$ to $\uparrow$ and $s_k^-$ lowers electron $k$ from $\uparrow$ to $\downarrow$.

**Diagonal terms** ($k = l$): $s_l^- s_l^+$ is the identity on each $\downarrow$ electron, contributing $+n_\downarrow$ total.

**Off-diagonal terms** ($k \ne l$, with $k \in \uparrow$ and $l \in \downarrow$): applying $s_l^+$ then $s_k^-$ swaps the spin labels of electrons $k$ and $l$. For an antisymmetric wavefunction whose first $n_\uparrow$ coordinates are spin-up, this is equivalent to swapping the spatial coordinates of electrons $k$ and $l$. Antisymmetry gives a factor of $-1$, which cancels the minus sign in the $-S_-S_+$ contribution, producing the positive ratios $\Psi(\mathbf{r}_{k\leftrightarrow l})/\Psi(\mathbf{r})$ in the formula below.

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

The sum over swaps uses a signed log-sum-exp for numerical stability.

## See also

- Configuration: [Molecule](#molecule-estimators)
- API: {class}`~jaqmc.estimator.spin.SpinSquared`
