# Extending JaQMC

Write custom workflows, wavefunctions, and configurable components. If you are comfortable with Python but new to JAX, read <project:jax-for-jaqmc.md> before diving into the extension guides.

## I Want To...

- Understand how walker data is structured and batched → <project:runtime-data-conventions.md>
- Add a new system/app workflow (new `jaqmc <app> train` / `evaluate` path) → <project:writing-workflows.md>
- Add a new wavefunction architecture → <project:wavefunctions.md>
- Add a new estimator, sampler, optimizer, or writer → <project:custom-components/index.md>
- Understand programmatic configuration and module loading → <project:configuration.md>
- Contribute changes back to the project → <project:contributing.md>

If you are starting from scratch, the usual reading order is
<project:runtime-data-conventions.md> -> <project:writing-workflows.md> ->
<project:wavefunctions.md>, then the custom-component guides you actually need.

## What You Can Extend

| What | Purpose | Example |
|------|---------|---------|
| Workflow | Assembles components for a particular system type | Molecule, solid, quantum Hall |
| Wavefunction | Maps electron positions to log\|psi\| | FermiNet, PsiFormer, Laughlin |
| Estimator | Computes a physical observable | Spin, electron density, Ewald energy |
| Sampler | Proposes and accepts/rejects electron moves | Metropolis-Hastings |
| Optimizer | Transforms gradients into parameter updates | KFAC, Optax wrappers |
| Writer | Records per-step statistics | Console, HDF5, CSV |

Most custom work involves estimators (new observables) or wavefunctions (new ansatzes). Samplers, optimizers, and writers rarely need custom implementations — the built-in options cover most use cases.

## Configure, Then Execute

A JaQMC run has two distinct phases.

In the **configure phase**, the workflow function creates components and connects them using a *stage builder*, which wires their runtime dependencies and produces a *work stage* ready to execute. This phase is fast — `--dry-run` stops here, letting you validate the entire setup without waiting for training.

In the **execute phase**, the training loop runs the assembled components once per iteration:

1. **Sample** — propose new electron positions via MCMC
2. **Evaluate** — run each estimator to compute observables
3. **Compute gradients** — differentiate the total energy with respect to parameters
4. **Update** — apply the optimizer

```{toctree}
:maxdepth: 2
:hidden:

jax-for-jaqmc.md
runtime-data-conventions.md
writing-workflows.md
wavefunctions.md
custom-components/index.md
configuration.md
contributing.md
```
