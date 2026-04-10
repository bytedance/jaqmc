# Samplers

The sampler draws electron configurations from $|\psi|^2$ using Markov Chain Monte Carlo (MCMC). See <project:../guide/sampling.md> for background on how MCMC sampling works and how to tune the sampler.

## Configuration

For sampler config keys, see the configuration reference: [Molecule](#train-sampler), [Solid](#solid-train-sampler), or [Hall](#hall-train-sampler).

## Protocols

```{eval-rst}
.. autoclass:: jaqmc.sampler.SamplerLike
   :members:

.. autoclass:: jaqmc.sampler.base.SamplerInit
   :special-members: __call__

.. autoclass:: jaqmc.sampler.base.SamplerStep
   :special-members: __call__

.. autoclass:: jaqmc.sampler.base.BatchLogProb
   :special-members: __call__

.. autoclass:: jaqmc.sampler.SamplePlan
   :members:
```

## Built-in samplers

```{eval-rst}
.. autoclass:: jaqmc.sampler.mcmc.MCMCSampler
   :members:
   :inherited-members:

.. autoclass:: jaqmc.sampler.mcmc.SamplingProposal
   :special-members: __call__

.. autoclass:: jaqmc.sampler.mcmc.MCMCState
```
