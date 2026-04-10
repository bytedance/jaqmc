# Wavefunction Architectures

JaQMC represents the many-electron wavefunction with a neural network ansatz. Several architectures are available, each with presets that trade expressiveness for compute cost. All wavefunction settings live under the `wf.*` config scope.

Start with **FermiNet** (the default) — it works well for most systems. For molecules with more than ~30 electrons, **Psiformer** can be more expressive but is more expensive per step.

```{tip}
To switch architectures, set `wf.module` — for example, `wf.module=psiformer` on the CLI. See [Swappable Modules](#swappable-modules) for how module paths work. To verify convergence, increase `ndets` or the hidden dimensions and check whether the final energy improves — if it doesn't, the smaller preset is sufficient.
```

## FermiNet

**Systems**: molecules, solids

FermiNet captures electron correlations through two parallel feature streams — one for each electron's interaction with atoms, another for explicit pairwise electron-electron features — that exchange information at each layer. The final features are projected into orbital matrices, multiplied by an envelope that enforces exponential decay away from atoms, and combined via a log-sum-of-determinants.

- **Paper**: [Ab initio solution of the many-electron Schrödinger equation with deep neural networks](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429)

### Presets

Small FermiNet (the default — shown here for reference, no overrides needed):

```yaml
wf:
  hidden_dims_single: [256, 256, 256, 256]
  hidden_dims_double: [32, 32, 32, 32]
  ndets: 16
```

Large FermiNet:

```yaml
wf:
  hidden_dims_single: [512, 512, 512, 512]
  hidden_dims_double: [32, 32, 32, 32]
  ndets: 32
```

## Psiformer

**Systems**: molecules

Psiformer replaces FermiNet's explicit pairwise stream with **multi-head self-attention**, capturing electron-electron correlations implicitly through attention rather than explicit features. This tends to be more accurate on larger molecules — in the original paper, a small Psiformer outperformed a large FermiNet despite having fewer parameters, with the advantage growing with system size.

As with FermiNet, the output features are projected into enveloped orbitals and combined via a log-sum-of-determinants. Psiformer additionally includes an electron-electron **Jastrow factor** that enforces the exact cusp condition (see the paper for details).

- **Paper**: [Self-Attention Ansatz for Ab-Initio Quantum Chemistry](https://openreview.net/forum?id=xveTeHVlF7j)

### Presets

Small Psiformer:

```yaml
wf:
  module: psiformer
  num_layers: 4
  num_heads: 4
  heads_dim: 64
  ndets: 16
```

Large Psiformer:

```yaml
wf:
  module: psiformer
  num_layers: 4
  num_heads: 8
  heads_dim: 64
  ndets: 32
```

## Full Configuration Reference

For the complete list of configurable fields, see the reference pages:

- [Molecule wavefunction options](#molecule-train-wf) — FermiNet and Psiformer fields
- [Solid wavefunction options](#solid-train-wf) — FermiNet with PBC extensions
- [API reference](../api-reference/wavefunctions.md) — Base classes, protocols, and output types

If you are implementing a new architecture rather than selecting built-ins, continue with
<project:../extending/wavefunctions.md>.
