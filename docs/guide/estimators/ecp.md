# Pseudopotentials (ECP)

For configuration options, see the estimator sections of each system's config reference ([molecule](#molecule-estimators), [solid](#solid-estimators)).

Effective core potentials (ECPs) replace the expensive all-electron treatment of core electrons with a smooth pseudopotential. Instead of explicitly simulating the core electrons, each ECP atom exerts an effective potential on nearby valence electrons that mimics the effect of the missing core. The ECP estimator computes the energy contribution from this potential — reported as `energy:ecp` in the training output.

## ECP Hamiltonian

The ECP adds the following terms to the Hamiltonian for each valence electron at distance $r$ from an ECP atom:

$$
\hat{V}_\text{ECP} = V_\text{loc}(r) + \sum_{l=0}^{l_\text{max}} V_l(r) \sum_{m=-l}^{l} |lm\rangle\langle lm|
$$

The first term $V_\text{loc}(r)$ is a **local** potential that depends only on the electron-atom distance. The second term is **nonlocal** — it projects the valence electron onto angular momentum channels $|lm\rangle$ (spherical harmonics centered at the atom), weights each channel by a radial function $V_l(r)$, and sums over channels. Each radial function is a tabulated sum of Gaussian-weighted powers:

$$
V_l(r) = \sum_k c_k \, r^{n_k - 2} \exp(-\alpha_k r^2)
$$

where the coefficients $c_k$, powers $n_k$, and exponents $\alpha_k$ are read from PySCF's ECP database (see Eq. 5 of [Li et al., Phys. Rev. Research 4, 013021 (2022)](https://doi.org/10.1103/PhysRevResearch.4.013021)). The same functional form is used for $V_\text{loc}$.

## Energy evaluation

The **local** energy is straightforward: evaluate $V_\text{loc}(r)$ at each valence electron-atom distance and sum.

The **nonlocal** energy requires evaluating the projectors $|lm\rangle\langle lm|$ on the wavefunction, which produces an angular integral. After aligning the polar axis with the electron-atom direction, the sum over $m$ simplifies to a Legendre polynomial ([Fahy, Wang & Louie, Phys. Rev. B 42, 3503 (1990)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.42.3503)), giving the contribution from channel $l$:

$$
E_l = V_l(r) \frac{2l+1}{4\pi} \int
  P_l(\cos\theta) \,
  \frac{\psi(\mathbf{r}')}{\psi(\mathbf{r})} \, d\Omega
$$

where $r = |\mathbf{r} - \mathbf{R}|$ is the electron-atom distance, $\mathbf{r}' = \mathbf{R} + r\,\hat{\Omega}$ is the electron displaced to direction $\hat{\Omega}$ on a sphere of radius $r$ centered at the atom, $\theta$ is the angle between $\mathbf{r}' - \mathbf{R}$ and $\mathbf{r} - \mathbf{R}$, and $P_l$ is the Legendre polynomial. The integral runs over all directions $\hat{\Omega}$ and is evaluated numerically using spherical quadrature.

## Computational details

**Nearest-core optimization.** For each electron, only the `max_core` nearest ECP atoms are considered for the nonlocal integral. This limits cost when many ECP atoms are present.

**Random quadrature rotation.** Each electron gets a randomly rotated copy of the quadrature grid, reducing systematic bias from a fixed grid orientation. This means the ECP energy contribution has stochastic noise even at fixed electron positions.

**PBC support.** When lattice vectors are provided, electron-atom distances use minimum-image convention, and displaced electrons are wrapped back into the cell with the appropriate Bloch phase (twist angle).

## See also

- The ECP estimator is automatically added when `ecp` is set in the system configuration. See [Basis Sets and ECPs](#molecule-basis-sets-and-ecps).
- Configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators)
- API: {class}`~jaqmc.estimator.ecp.estimator.ECPEnergy`
