# Ewald summation

For configuration options, see the [solid config reference](#solid-estimators).

The Ewald summation computes electrostatic (Coulomb) energies in periodic systems. Direct summation of $1/r$ interactions converges extremely slowly in periodic boundary conditions; the Ewald technique decomposes the sum into rapidly converging real-space and reciprocal-space series.

For a pedagogical derivation, see [Ewald Summation (Qijing Zheng)](http://staff.ustc.edu.cn/~zqj/posts/Ewald-Summation/).

## Formulation

The key idea is to split each point charge into a short-range part (screened by a Gaussian) and a long-range part (the compensating Gaussian). The short-range part converges quickly in real space; the smooth long-range part converges quickly in reciprocal (Fourier) space.

The total electrostatic energy is:

$$
V_\text{Ewald} = V_\text{real} + V_\text{recip} + V_\text{self} + V_\text{charged}
$$

All formulas below use atomic units ($4\pi\epsilon_0 = 1$). $\Omega$ denotes the simulation cell volume and $\mathbf{n}$ runs over lattice translation vectors (periodic images). The implementation treats all charged particles (electrons and ions) uniformly.

### Real-space term

Interactions between screened charges, summed over particle pairs $i, j$ and periodic images $\mathbf{n}$. The self-interaction ($i=j$, $\mathbf{n}=\mathbf{0}$) is excluded:

$$
V_\text{real} = \frac{1}{2} \sum_{\mathbf{n}} \sum_{i, j} q_i q_j
  \frac{\text{erfc}(\alpha |\mathbf{r}_{ij} + \mathbf{n}|)}
  {|\mathbf{r}_{ij} + \mathbf{n}|}
  \times (1 - \delta_{\mathbf{n},0}\delta_{ij})
$$

### Reciprocal-space term

The compensating Gaussian distributions are summed in Fourier space over reciprocal lattice vectors $\mathbf{G} \neq 0$:

$$
V_\text{recip} = \sum_{\mathbf{G} \neq 0} W(G) |S(\mathbf{G})|^2
$$

where:

$$
W(G) = \frac{4\pi}{\Omega G^2} e^{-G^2 / 4\alpha^2}, \qquad
S(\mathbf{G}) = \sum_{k} q_k e^{i \mathbf{G} \cdot \mathbf{r}_k}
$$

### Self-energy correction

Removes the spurious interaction of each Gaussian cloud with its own point charge:

$$
V_\text{self} = - \frac{\alpha}{\sqrt{\pi}} \sum_k q_k^2
$$

### Charged-system correction

For non-neutral cells, the $\mathbf{G}=0$ divergence is regularized by a uniform neutralizing background:

$$
V_\text{charged} = - \frac{\pi}{2 \Omega \alpha^2} \left( \sum_k q_k \right)^2
$$

## Computational details

**Ewald parameter $\alpha$.** Chosen heuristically as $\alpha = 5.0 / h_\text{min}$, where $h_\text{min}$ is the smallest perpendicular height of the simulation cell. This ensures the real-space erfc terms decay to negligible values within the cell boundaries.

**G-vector selection.** Candidate reciprocal lattice vectors are generated on a grid bounded by `ewald_gmax`. Only vectors whose weight $W(G)$ exceeds a tolerance ($10^{-12}$) are kept, which dramatically reduces the sum size.

## See also

- The Ewald parameters are not user-configurable — they are set internally by the
  solid workflow's potential-energy estimator (`ewald_gmax=200`, `nlatvec=1`).
- API: {class}`~jaqmc.estimator.ewald.EwaldSum`
