# Roadmap

JaQMC's near-term direction is to make advanced capabilities fit naturally into the same modular workflow model that already powers current molecule, solid, and quantum Hall workflows. Rather than growing a long list of isolated features, the focus is on turning useful legacy and system-specific ideas into reusable, first-class framework components.

In practice, that means investing in estimator infrastructure, Hamiltonian support, and training objectives that broaden what can be expressed without forcing users onto special-case code paths.

This page is informational: it highlights the areas JaQMC is currently evolving toward. It is not a public task list, contribution roadmap, or commitment to a specific timeline or implementation order. The directions below reflect current project emphasis and may be useful context for contributors.

## Legacy Feature Migration

We will gradually integrate features from the previous version of JaQMC into the current framework.

- **Sparse forward Laplacian support**
  Improve the current forward Laplacian handling to support baking sparsity information directly into the network definition.
  *Ref:* [Forward Laplacian](https://www.nature.com/articles/s42256-024-00794-x).

- **Diffusion Monte Carlo**
  *Ref:* [NNDMC](https://www.nature.com/articles/s41467-023-37609-3).

- **Pseudo Hamiltonian support**
  Bring pseudopotential-related machinery into the current framework as first-class, configurable support rather than legacy add-ons.
  *Ref:* [NNQMC-PH](https://arxiv.org/abs/2505.19909).

- **Spin- and overlap-aware training penalties**
  Modernize useful legacy penalty mechanisms so they integrate cleanly with today's training stack.
  *Ref:* [Spin-plus Penalty](https://www.nature.com/articles/s43588-024-00730-4).



## Long-term Roadmap

We aim to continuously improve JaQMC, with efforts focused on, but not limited to, the following areas:

- **Neural Network Ansatz**

- **Optimization Methods**

- **Systems and Hamiltonians**


- **Reusable Estimators**

- **Other Cutting-edge Techiniques**
