# Atomic data

Reference for JaQMC's shared atomic-system APIs, from `Element` and `Atom`
through `AtomicSystemConfig`, SCF/orbital helpers, electron initialization,
and pretraining utilities.

## Elements and atoms

JaQMC builds molecule and solid atomic systems from four core types.
{class}`~jaqmc.utils.atomic.elements.Element` stores periodic-table metadata
such as atomic number and neutral-atom `unpaired_electron` defaults.
{class}`~jaqmc.utils.atomic.atom.Atom` stores one resolved atom with Cartesian
coordinates and an effective charge.

{class}`~jaqmc.utils.atomic.atomic_system.AtomicSystemConfig` combines those
atoms with system-wide charge and spin settings. Its main fields are `pp`
(which pseudopotential treatment, if any, is used when resolving explicit
electron counts), `total_charge` (the net charge applied during that electron
counting), and `s_z` (the total spin projection used when splitting the
explicitly simulated electrons into spin-up and spin-down counts). From those
inputs, JaQMC derives values such as `electron_spins`.

{class}`~jaqmc.utils.atomic.atomic_system.AtomInitialization` stores optional
per-atom hints used only when seeding initial electron positions.

For worked examples of `pp`, `total_charge`, `s_z`, and the resulting electron
counts, see the molecule overview (<project:../systems/molecule/index.md>) and
the solid overview (<project:../systems/solid/index.md>).

```{eval-rst}
.. autoclass:: jaqmc.utils.atomic.elements.Element
   :members:

.. autoclass:: jaqmc.utils.atomic.atom.Atom
   :members:

.. autoclass:: jaqmc.utils.atomic.atomic_system.AtomicSystemConfig
   :members:

.. autoclass:: jaqmc.utils.atomic.atomic_system.AtomInitialization
   :members:
```

## System-specific atom config schemas

The shared atomic layer does not define a generic YAML `AtomConfig`. Concrete
atom-entry schemas live in the molecule and solid config layers on top of
{class}`~jaqmc.utils.atomic.atomic_system.AtomicSystemConfig`.

This section documents those entry types. For config keys, defaults, and full
workflow usage, see the molecule configuration reference
(<project:../systems/molecule/train.md>) and the solid configuration reference
(<project:../systems/solid/train.md>).

Use {class}`~jaqmc.app.molecule.config.base.AtomConfig` for molecules with
Cartesian coordinates in the enclosing system unit, and
{class}`~jaqmc.app.solid.config.base.SolidAtomConfig` for solids with
primitive-cell fractional coordinates.

```{eval-rst}
.. autoclass:: jaqmc.app.molecule.config.base.AtomConfig
   :members:

.. autoclass:: jaqmc.app.solid.config.base.SolidAtomConfig
   :members:
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

## Pseudopotentials

The unified `pp` vocabulary and public helper functions are re-exported from
`jaqmc.utils.atomic`.

```{eval-rst}
.. autodata:: jaqmc.utils.atomic.PP_PH
.. autodata:: jaqmc.utils.atomic.SUPPORTED_PH_ELEMENTS
.. autofunction:: jaqmc.utils.atomic.core_electrons_by_pp
```

JaQMC also exposes `jaqmc.utils.atomic.PH_SURROGATE_ECP`, the internal mapping
from each PH-supported element to the ECP used to bootstrap the SCF pretrain.

## Electron initialization

```{eval-rst}
.. autofunction:: jaqmc.utils.atomic.initialization.distribute_spins
.. autofunction:: jaqmc.utils.atomic.initialization.initialize_electrons_gaussian
```

## Pretraining

```{eval-rst}
.. autoclass:: jaqmc.utils.atomic.pretrain.PretrainReferenceConfig

.. autofunction:: jaqmc.utils.atomic.pretrain.make_pretrain_log_amplitude
.. autofunction:: jaqmc.utils.atomic.pretrain.make_pretrain_loss
```
