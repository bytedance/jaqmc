# Density

For configuration options, see the estimator sections of each system's config reference ([molecule](#molecule-estimators), [solid](#solid-estimators), [hall](#hall-estimators)).

The density estimator accumulates histograms of electron positions over evaluation steps, producing a discretized picture of the electron density $n(\mathbf{r})$. Three variants match the three geometry types in JaQMC:

- {class}`~jaqmc.estimator.density.CartesianDensity` — projects positions onto user-defined directions in Cartesian space.  Suited to molecules and other open-boundary systems.
- {class}`~jaqmc.estimator.density.FractionalDensity` — converts positions to fractional (lattice) coordinates and histograms within $[0, 1)$.  Suited to periodic solids.
- {class}`~jaqmc.estimator.density.SphericalDensity` — histograms polar angle $\theta$ (and optionally azimuthal angle $\varphi$) on the Haldane sphere.  Suited to FQHE simulations.

All three are disabled by default and enabled via `estimators.enabled.density: true`.

## How it works

Each variant inherits from `HistogramEstimator`, which handles the accumulation loop. On every evaluation step the estimator:

1. Calls `extract(data)` to map electron positions to histogram coordinates (a method each variant implements differently).
2. Bins the extracted coordinates into an N-dimensional histogram using `jnp.histogramdd`.
3. Accumulates the bin counts into a running total using Kahan summation to maintain float32 precision over millions of steps.

The result is a raw count histogram stored in the estimator state under `"histogram"`. Normalization (dividing by step count, bin volume, or particle number) is left to post-processing.

## Configuring axes

For `CartesianDensity` and `FractionalDensity`, each active entry in the
`axes` dictionary defines one histogram dimension. One active axis gives a 1-D
histogram, two active axes give a 2-D histogram, and three active axes give a
3-D histogram.

The estimator classes themselves do not supply a fixed set of default axes.
Instead, each workflow chooses defaults that match its geometry. The molecule
and solid evaluation workflows wire three axes by default (`x`/`y`/`z` for
molecules, `a`/`b`/`c` for solids), while other workflows may choose a
different default configuration.

This example keeps only the `z` axis active:

```yaml
estimators:
  enabled:
    density: true
  density:
    axes:
      z:
        direction: [0, 0, 1]   # project onto z-axis (normalized internally)
        bins: 100               # number of histogram bins
        range: [-15.0, 15.0]   # min/max bounds in bohr
```

### Disabling workflow-provided default axes

Because the config system deep-merges user overrides into workflow defaults,
specifying only one axis does not remove the others. For example, overriding
only `z` in the molecule evaluation workflow still leaves the default `x` and
`y` axes active. To keep a subset, set the unwanted ones to `null`. This works
for both `CartesianDensity` and `FractionalDensity`:

```yaml
estimators:
  density:
    axes:
      x: null  # remove x
      y: null  # remove y
      # z is inherited from the workflow default
```

Entries set to `null` are filtered out before histogram construction.

## See also

- Configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators), [Hall](#hall-estimators)
- API: {class}`~jaqmc.estimator.density.CartesianDensity`, {class}`~jaqmc.estimator.density.FractionalDensity`, {class}`~jaqmc.estimator.density.SphericalDensity`
