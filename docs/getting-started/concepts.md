# Core Concepts

This page explains how JaQMC works: what happens during a simulation, what the main components are, and the vocabulary you'll encounter throughout the documentation.

## What a JaQMC Run Does

JaQMC solves quantum many-body problems by optimizing a neural-network wavefunction. The primary method is **Variational Monte Carlo (VMC)**, which targets ground states using the variational principle: any trial wavefunction gives an energy that is an upper bound on the true ground-state energy. So the optimizer's job is simple — push the energy down. In practice, the per-step energy is noisy, so treat plateaus as a convergence signal only after checking evaluation statistics and uncertainty estimates.

In practice, a VMC training run repeats the same cycle every step:

1. **Sample** electron positions from the current wavefunction
2. **Evaluate** the local energy at each sampled position
3. **Compute** parameter gradients from those energies
4. **Update** the wavefunction parameters

This cycle continues for a fixed number of steps. The energy estimate should decrease over time and stabilize.

## The VMC Training Loop

The VMC training loop is built from four independent components. Each one handles a specific part of the cycle, and each can be swapped via <project:../guide/configuration.md> without touching the others.

```{graphviz}
digraph {
    layout=circo
    node [shape=box, style=rounded]

    E [label=<
      <table border="0" cellspacing="0" cellpadding="2">
        <tr><td><b>Estimators</b></td></tr>
        <tr><td>e.g. local energy</td></tr>
      </table>
    >]
    O [label=<
      <table border="0" cellspacing="0" cellpadding="2">
        <tr><td><b>Optimizer</b></td></tr>
        <tr><td>e.g. KFAC, Adam</td></tr>
      </table>
    >]
    W [label=<
      <table border="0" cellspacing="0" cellpadding="2">
        <tr><td><b>Wavefunction</b></td></tr>
        <tr><td>e.g. FermiNet</td></tr>
      </table>
    >]
    S [label=<
      <table border="0" cellspacing="0" cellpadding="2">
        <tr><td><b>Sampler</b></td></tr>
        <tr><td>e.g. MCMC</td></tr>
      </table>
    >]

    W -> S [headlabel="sample from |ψ|²", labeldistance=4, labelangle=-70]
    S -> E [taillabel="electron positions", labeldistance=4, labelangle=70]
    E -> O [headlabel="energy gradient", labeldistance=4, labelangle=-70]
    O -> W [xlabel="updated params"]
}
```

### Wavefunction

The wavefunction is a neural network that takes electron positions and outputs a log-amplitude. It encodes everything the model "knows" about the quantum state — the optimizer adjusts its parameters to lower the energy.

JaQMC ships with several architectures, including [FermiNet](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429) (the default) and [Psiformer](https://openreview.net/forum?id=xveTeHVlF7j) (attention-based), all of which satisfy the antisymmetry requirement of fermionic wavefunctions by construction. See <project:../guide/wavefunction.md> for architecture details and presets.

### Sampler

Computing the energy exactly would require integrating over all possible electron positions — intractable for more than a few electrons. Instead, the sampler draws representative electron positions from the probability distribution $|\psi|^2$ using **Markov Chain Monte Carlo (MCMC)**.

Each independent MCMC chain is called a **walker**. At every step, the sampler proposes a move for each walker's electrons and accepts or rejects it based on the wavefunction. The fraction of accepted moves is the **pmove** — values around 0.5 indicate healthy sampling. If pmove is too high, walkers aren't exploring enough; too low, and most proposals are wasted.

### Estimators

Given a batch of sampled electron positions, estimators compute physical quantities. The most important is the **local energy**:

$$
E_L(\mathbf{r}) = \frac{\hat{H}\,\psi(\mathbf{r})}{\psi(\mathbf{r})}
$$

where $\hat{H}$ is the Hamiltonian and $\mathbf{r}$ is an electron configuration. The local energy is evaluated at each walker position, and its mean over walkers gives the variational energy estimate — the number reported as `energy` in the training output.

Estimators also compute individual energy components (kinetic, electron-electron, electron-ion) and can compute non-energy observables like $\langle S^2 \rangle$. Multiple estimators run in a pipeline, and their outputs are written to the training statistics files. See <project:../guide/estimators/index.md> for details.

### Optimizer

The optimizer updates the wavefunction parameters to minimize the energy. JaQMC supports any optimizer from [Optax](inv:optax:*:doc#index) (Adam, LAMB, etc.) and [KFAC](https://arxiv.org/abs/1503.05671) (a second-order optimizer that uses curvature information from the wavefunction). You can also plug in your own. KFAC typically converges faster for VMC but is more expensive per step.

## What to Watch During Training

When a run is working well:

- **Energy** (`energy`) decreases over the first portion of training, then stabilizes. The stable value should be close to reference values for well-studied systems.
- **pmove** stays in the acceptable range 0.3–0.7 (the sampler auto-tunes toward 0.50–0.55). Persistent drift outside this range suggests the sampler step size needs adjustment.
- **Energy variance** decreases as the wavefunction improves. High variance means the wavefunction is a poor fit in some regions of configuration space.

If you see energy increasing, pmove collapsing, or NaN values, see <project:../guide/troubleshooting.md>.
