# Atomic data

Atom definitions, element properties, electronic structure (SCF), orbital evaluation, electron initialization, and pretraining utilities.

## Elements and atoms

```{eval-rst}
.. autoclass:: jaqmc.utils.atomic.elements.Element
   :members:

.. autoclass:: jaqmc.utils.atomic.atom.Atom

.. autoclass:: jaqmc.utils.atomic.atomic_system.AtomicSystemConfig
```

## Self-consistent field (SCF)

```{eval-rst}
.. autoclass:: jaqmc.utils.atomic.scf.MolecularSCF
   :members:

.. autoclass:: jaqmc.utils.atomic.scf.PeriodicSCF
   :members:
```

## Gaussian-type orbitals

```{eval-rst}
.. autoclass:: jaqmc.utils.atomic.gto.AtomicOrbitalEvaluator
   :members:

.. autoclass:: jaqmc.utils.atomic.gto.PBCAtomicOrbitalEvaluator
   :members:

.. autofunction:: jaqmc.utils.atomic.gto.solid_harmonic
.. autofunction:: jaqmc.utils.atomic.gto.cart2sph
```

## Effective core potentials

```{eval-rst}
.. autofunction:: jaqmc.utils.atomic.ecp.get_valence_spin_config
.. autofunction:: jaqmc.utils.atomic.ecp.get_core_electrons
```

## Electron initialization

```{eval-rst}
.. autofunction:: jaqmc.utils.atomic.initialization.distribute_spins
.. autofunction:: jaqmc.utils.atomic.initialization.initialize_electrons_gaussian
```

## Pretraining

```{eval-rst}
.. autofunction:: jaqmc.utils.atomic.pretrain.make_pretrain_log_amplitude
.. autofunction:: jaqmc.utils.atomic.pretrain.make_pretrain_loss
```
