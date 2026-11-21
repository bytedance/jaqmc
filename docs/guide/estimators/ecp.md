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

**Spherical quadrature rule.** `quadrature_id` accepts `octahedron_6`,
`octahedron_18`, `octahedron_26`, `octahedron_50`, `icosahedron_12`, or
`icosahedron_32`. The suffix is the number of integration points. Omitting the
option, or setting it to `null`, uses `icosahedron_12`.

**PBC support.** When lattice vectors are provided, electron-atom distances use minimum-image convention, and displaced electrons are wrapped back into the cell with the appropriate Bloch phase (twist angle).

(ecp-nonlocal-precision)=
### Nonlocal precision and tiered pair schedules

Each nonlocal channel evaluates the wavefunction at every quadrature point for every considered $(e, \text{core})$ pair. When the active wavefunction supports reduced-precision modes (see <project:../mixed-precision.md>), `estimators.energy.ecp.wavefunction_precision` can accelerate these evaluations in two ways:

- **Single mode**: A mode string such as `fp16_mixed` evaluates all nonlocal quadrature points in that mode (`null` inherits `wf.precision`).
- **Rank-fraction tier mapping**: A mapping of `{mode: [low, high]}` with $0 \le \text{low} < \text{high} \le 1$ sorts the $N_{\text{elec}} \times \text{max\_core}$ electron-core pairs in descending order of nonlocal radial magnitude $\max_{l \ge 0} |V_l(r)|$ (from strongest at `0.0` to weakest at `1.0`) and assigns each non-overlapping rank-fraction slice to the specified mode. The key `default` inherits `wf.precision`:

  ```yaml
  estimators:
    energy:
      ecp:
        wavefunction_precision:
          default: [0.0, 0.5]
          fp16_mixed: [0.5, 1.0]
  ```

  This evaluates the strongest 50% (`[0.0, 0.5]`) of electron-core pairs at the wavefunction's default precision and the weakest 50% (`[0.5, 1.0]`) in `fp16_mixed`. To evaluate all considered pairs without extra truncation beyond `max_core`, the configured intervals must cover `[0.0, 1.0]`: any pair outside all `[low, high]` intervals is skipped. When running `evaluate` in standard precision with a shared YAML file that defines a tier mapping, pass `estimators.energy.ecp.wavefunction_precision=null` to reset it (see [Verify final observables in standard precision](#mixed-precision-evaluation-and-resuming)).

## See also

- In the molecule and solid workflows, `estimators.energy.ecp` is enabled
  automatically when `system.pp` selects an ECP for at least one atom.
- Because `system.pp` also changes the explicit electron count, arbitrary
  molecule and solid configs should set `system.s_z` and optional
  `system.total_charge` and let JaQMC derive the spin-up and spin-down counts.
  See [Pseudopotentials](#molecule-pseudopotentials).
- Configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators)
- API: {class}`~jaqmc.estimator.ecp.estimator.ECPEnergy`
