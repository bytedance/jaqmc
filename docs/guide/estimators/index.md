# How Estimators Work

Physics, derivations, and computational details behind JaQMC's built-in estimators. For configuration options, see the training ([molecule](#train-estimators), [solid](#solid-train-estimators), [hall](#hall-train-estimators), [moire](#moire-train-estimators), [electron gas](#electron-gas-config)) and evaluation ([molecule](#molecule-estimators), [solid](#solid-estimators), [hall](#hall-estimators), [moire](#moire-estimators)) estimator sections of each system's config reference.

- <project:kinetic.md> — Euclidean kinetic energy, plus the spherical variant for FQHE
- <project:angular-momentum.md> — Angular momentum and sector projection on the Haldane sphere
- <project:ewald.md> — Coulomb energy in periodic systems via real/reciprocal-space decomposition
- <project:ecp.md> — Semi-local effective core potential (ECP) contributions, quadrature details
- <project:ph.md> — Local pseudopotential, or Pseudo-Hamiltonian (PH), parallel to the semi-local ECP family
- <project:spin.md> — Spin contamination measurement and penalty via coordinate-swap ratios
- <project:overlap.md> — Wavefunction overlap estimation and orthogonalization penalty for excited states
- <project:density.md> — Electron density histograms (Cartesian, fractional, and spherical)
- <project:loss-grad.md> — VMC gradient estimator, penalty loss terms, and outlier clipping

If you are implementing new estimators rather than tuning built-ins, continue with
<project:/extending/custom-components/estimators.md>. For protocol and class-level details, see
<project:/api-reference/estimators.md>.

```{toctree}
:hidden:

kinetic.md
angular-momentum.md
ewald.md
ecp.md
ph.md
spin.md
overlap.md
density.md
loss-grad.md
```
