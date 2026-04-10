# Optimizers

JaQMC provides natural gradient optimizers (KFAC, SR) alongside standard optimizers from Optax. Natural gradient methods update the wavefunction in the Hilbert space rather than parameter space, which is typically more stable for VMC. See <project:../guide/optimizers/index.md> for background on choosing an optimizer.

## Configuration

For optimizer config keys, see the configuration reference: [Molecule](#train-optim), [Solid](#solid-train-optim), or [Hall](#hall-train-optim).

## Protocol

```{eval-rst}
.. autoclass:: jaqmc.optimizer.base.OptimizerLike
   :members:

.. autoclass:: jaqmc.optimizer.base.OptimizerInit
   :special-members: __call__

.. autoclass:: jaqmc.optimizer.base.OptimizerUpdate
   :special-members: __call__
```

## Optimizers provided by JaQMC

```{eval-rst}
.. autoclass:: jaqmc.optimizer.sr.SROptimizer
.. autoclass:: jaqmc.optimizer.kfac.kfac.KFACOptimizer
```

## Optimizers provided by <inv:optax:*:doc#index>

```{note}
When using Optax, always use the ``optax:<name>`` wrapper (e.g. ``optax:adam``) to ensure compatibility with JaQMC's configuration system.
```

You can find the full list of optimizers in [Optax documentation](inv:optax:*:doc#api/optimizers).

```{eval-rst}
.. autoclass:: jaqmc.optimizer.optax.adam
.. autoclass:: jaqmc.optimizer.optax.lamb
```

## Learning rate schedules

```{eval-rst}
.. autoclass:: jaqmc.optimizer.schedule.Standard
.. autoclass:: jaqmc.optimizer.schedule.Constant
```
