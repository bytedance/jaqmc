# Estimators

API reference for built-in estimators. For background, formulas, and configuration guidance, see the <project:/guide/estimators/index.md>.

## Base classes

```{eval-rst}
.. autoclass:: jaqmc.estimator.Estimator
   :members: init, evaluate_batch_walkers, reduce, finalize_stats, finalize_state

.. autoclass:: jaqmc.estimator.PerWalkerEstimator
   :members: evaluate_single_walker, evaluate_batch_walkers

.. autoclass:: jaqmc.estimator.FunctionEstimator
   :members:

.. autoclass:: jaqmc.estimator.EstimatorPipeline
   :members: init, evaluate, finalize_stats, digest

.. autotype:: jaqmc.estimator.EstimatorLike

.. autotype:: jaqmc.estimator.EstimateFn
```

## Built-in estimators

### Kinetic energy

```{eval-rst}
.. autoclass:: jaqmc.estimator.kinetic.EuclideanKinetic
.. autoclass:: jaqmc.estimator.kinetic.SphericalKinetic
.. autoclass:: jaqmc.estimator.kinetic.LaplacianMode
```

### Ewald summation

```{eval-rst}
.. autoclass:: jaqmc.estimator.ewald.EwaldSum
   :members: energy
```

### ECP energy

```{eval-rst}
.. autoclass:: jaqmc.estimator.ecp.estimator.ECPEnergy
```

### Spin squared

```{eval-rst}
.. autoclass:: jaqmc.estimator.spin.SpinSquared
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
.. autoclass:: jaqmc.estimator.loss_grad.LossAndGrad
```
