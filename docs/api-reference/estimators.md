# Estimators

API reference for built-in estimators. For background, formulas, and configuration guidance, see the <project:/guide/estimators/index.md>.

## Base classes

```{eval-rst}
.. autoclass:: jaqmc.estimator.Estimator
   :members: init, evaluate_batch_walkers, reduce, finalize_stats, finalize_state

.. autoclass:: jaqmc.estimator.PerWalkerEstimator
   :members: evaluate_single_walker, evaluate_batch_walkers

.. autoclass:: jaqmc.estimator.EstimatorCollection
   :members: is_enabled, set_enabled

.. autofunction:: jaqmc.estimator.WithToggles

.. autofunction:: jaqmc.estimator.HideLoss

.. autoclass:: jaqmc.estimator.EstimatorPipeline
   :members: init, evaluate, finalize_stats, digest

```

## Built-in estimators

### Kinetic energy

```{eval-rst}
.. autoclass:: jaqmc.estimator.kinetic.EuclideanKinetic
.. autoclass:: jaqmc.estimator.kinetic.SphericalKinetic
.. autoclass:: jaqmc.estimator.kinetic.LaplacianMode
```

### Angular momentum

```{eval-rst}
.. autoclass:: jaqmc.estimator.angular_momentum.SphericalAngularMomentum
.. autoclass:: jaqmc.estimator.angular_momentum.AngularMomentumLoss
.. autoclass:: jaqmc.estimator.angular_momentum.SphericalAngularMomentumProjection
```

### Ewald summation

```{eval-rst}
.. autoclass:: jaqmc.estimator.ewald.EwaldSum
   :members: energy
```

### ECP energy

```{eval-rst}
.. autoclass:: jaqmc.estimator.ecp.estimator.ECPEnergy

.. autoclass:: jaqmc.estimator.ecp.quadrature.ECPQuadrature
   :members:
   :undoc-members:
```

| `ECPQuadrature` value | Integration points |
|---|---:|
| `octahedron_6` | 6 |
| `octahedron_18` | 18 |
| `octahedron_26` | 26 |
| `octahedron_50` | 50 |
| `icosahedron_12` | 12 |
| `icosahedron_32` | 32 |

### PH energy

```{eval-rst}
.. autoclass:: jaqmc.estimator.ph.PHEnergy
```

### Spin

```{eval-rst}
.. autoclass:: jaqmc.estimator.spin.SpinSquared
```

### Overlap

```{eval-rst}
.. autoclass:: jaqmc.estimator.overlap.Overlap
   :members: restore
.. autoclass:: jaqmc.estimator.overlap.ReferenceState
```

### Total energy

```{eval-rst}
.. autoclass:: jaqmc.estimator.total_energy.TotalEnergy
```

### Density

```{eval-rst}
.. autoclass:: jaqmc.estimator.density.CartesianDensity
.. autoclass:: jaqmc.estimator.density.CartesianAxis
.. autoclass:: jaqmc.estimator.density.cartesian.CartesianAxis
   :members:
.. autoclass:: jaqmc.estimator.density.FractionalDensity
.. autoclass:: jaqmc.estimator.density.FractionalAxis
.. autoclass:: jaqmc.estimator.density.fractional.FractionalAxis
   :members:
.. autoclass:: jaqmc.estimator.density.SphericalDensity
```

### Loss and gradient

```{eval-rst}
.. autofunction:: jaqmc.estimator.loss_grad.loss_term
.. autoclass:: jaqmc.estimator.loss_grad.LossGrad
   :members:
```
