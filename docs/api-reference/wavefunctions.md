# Wavefunctions

A wavefunction maps electron configurations to log-amplitudes (and optionally signs or phases). All JaQMC wavefunctions are [Flax Linen modules](inv:flax:py:class#flax.linen.Module) that subclass {class}`~jaqmc.wavefunction.Wavefunction`.

For architecture-specific options (FermiNet, Psiformer, etc.), see the configuration reference pages ([Molecules](../systems/molecule/train.md), [Solids](../systems/solid/train.md), [Quantum Hall](../systems/hall/train.md)).

## Base class and protocols

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.Wavefunction
   :members: init_params, evaluate, __call__

.. autoclass:: jaqmc.wavefunction.WavefunctionLike
   :members:

.. autoclass:: jaqmc.wavefunction.WavefunctionEvaluate
   :special-members: __call__

.. autoclass:: jaqmc.wavefunction.WavefunctionInit
   :special-members: __call__

.. autotype:: jaqmc.wavefunction.base.NumericWavefunctionEvaluate
```

## Output types

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.base.RealWFOutput
.. autoclass:: jaqmc.wavefunction.base.ComplexWFOutput
.. autoclass:: jaqmc.wavefunction.base.LogPsiWFOutput
```

### Log-determinant output (used by FermiNet / Psiformer)

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.output.logdet.RealLogDetOutput
.. autoclass:: jaqmc.wavefunction.output.logdet.ComplexLogDetOutput
```

## Input features

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.input.atomic.AtomicEmbedding
.. autoclass:: jaqmc.wavefunction.input.atomic.MoleculeFeatures
   :members:
.. autoclass:: jaqmc.wavefunction.input.atomic.SolidFeatures
   :members:
```

## Backbone architectures

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.backbone.ferminet.FermiLayers
   :members:
.. autoclass:: jaqmc.wavefunction.backbone.psiformer.PsiformerBackbone
   :members:
.. autoclass:: jaqmc.wavefunction.backbone.psiformer.PsiformerLayer
   :members:
.. autoclass:: jaqmc.wavefunction.backbone.psiformer.LayerNormMode
```

## Orbital projection and envelope

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.output.orbital.OrbitalProjection
   :members:
.. autoclass:: jaqmc.wavefunction.output.orbital.SplitChannelDense
   :members:
.. autoclass:: jaqmc.wavefunction.output.envelope.Envelope
   :members:
.. autoclass:: jaqmc.wavefunction.output.envelope.EnvelopeType
.. autoclass:: jaqmc.wavefunction.output.logdet.LogDet
   :members:
```

## Jastrow factor

```{eval-rst}
.. autoclass:: jaqmc.wavefunction.jastrow.SimpleEEJastrow
   :members:

.. autoclass:: jaqmc.app.molecule.wavefunction.psiformer.JastrowType
```

(api-wavefunctions-data)=
## Data

In most wavefunction and per-walker estimator hooks,
{class}`~jaqmc.data.Data` represents one walker's structured runtime input.
{class}`~jaqmc.data.BatchedData` pairs a `Data`-shaped pytree with metadata
describing which fields carry a leading walker axis during batched execution.

Most user-defined wavefunctions and per-walker estimators only work with
`Data`. `BatchedData` is the lower-level representation used when framework code
or workflow plumbing needs to manipulate full walker batches explicitly.

For the built-in data-shape convention and the detailed explanation of
`fields_with_batch`, see <project:../extending/runtime-data-conventions.md>.

```{eval-rst}
.. autoclass:: jaqmc.data.Data
   :members: field_names, subset, merge

.. autoclass:: jaqmc.data.BatchedData
   :members:
```
