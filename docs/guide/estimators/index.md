# How Estimators Work

Physics, derivations, and computational details behind JaQMC's built-in estimators. For configuration options, see the estimator sections of each system's config reference ([molecule](#molecule-estimators), [solid](#solid-estimators), [hall](#hall-estimators)).

- <project:kinetic.md> — Laplacian computation, mode trade-offs, spherical variant for FQHE
- <project:ewald.md> — Coulomb energy in periodic systems via real/reciprocal-space decomposition
- <project:ecp.md> — Local and nonlocal core potential contributions, quadrature details
- <project:spin.md> — Spin contamination measurement via coordinate-swap ratios
- <project:density.md> — Electron density histograms (Cartesian, fractional, and spherical)
- <project:loss-grad.md> — Gradient estimator and outlier clipping

If you are implementing new estimators rather than tuning built-ins, continue with
<project:/extending/custom-components/estimators.md>. For protocol and class-level details, see
<project:/api-reference/estimators.md>.

```{toctree}
:hidden:

kinetic.md
ewald.md
ecp.md
spin.md
density.md
loss-grad.md
```
