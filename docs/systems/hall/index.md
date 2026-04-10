# Quantum Hall

## Background

In the fractional quantum Hall effect (FQHE), electrons confined to two dimensions under a strong magnetic field form strongly correlated states at fractional Landau level fillings (e.g., $\nu = 1/3$). Traditional numerical methods typically restrict electrons to the lowest Landau level (LLL), but in real systems the Coulomb interaction mixes higher Landau levels — an effect called **Landau level mixing** (LLM). Neural network wavefunctions in real space can naturally capture contributions from all Landau levels, overcoming this limitation.

The **Haldane sphere** is a compact geometry that places electrons on the surface of a sphere with a magnetic monopole at its center, producing a uniform radial field. The total magnetic flux through the sphere is $2Q\phi_0$ (where $2Q$ is an integer). This geometry has no edges and gives a well-defined filling factor $\nu$ via the relation $2Q = N/\nu - \mathcal{S}$, where $N$ is the electron count and $\mathcal{S}$ is a topological shift characteristic of each FQH state.

For the underlying physics, method details, and benchmarks, see [Qian et al., "Taming Landau level mixing in fractional quantum Hall states with deep learning" (arXiv:2412.14795)](https://arxiv.org/abs/2412.14795). For a comprehensive introduction to the FQHE and composite fermions, see [Jain, *Composite Fermions*, Cambridge University Press, 2007](https://doi.org/10.1017/CBO9780511607561).

## Basic Usage

The `jaqmc hall train` command runs VMC simulations on the Haldane sphere. The system is defined by the number of electrons and the magnetic flux:

```bash
# 3 spin-up electrons at flux 2Q = 6 (Laughlin 1/3 state)
jaqmc hall train system.nspins='[3,0]' system.flux=6

# Longer training run
jaqmc hall train ... train.run.iterations=10000
```

The default train preset is production-oriented. For shared quick-run workflow mechanics,
see [shared fast debug run](#recipe-fast-debug-run).

The sphere radius defaults to `sqrt(Q)` where `Q = flux / 2`. Override it with
`system.radius`. For the contextual defaults used by training, see the
[training configuration reference](#hall-train-system).

## How It Works

Each electron moves on the sphere surface in spherical coordinates `(theta, phi)`. The wavefunction ansatz is the **Monopole Harmonics Product Orbital (MHPO)** network.

1. Spherical coordinates are converted to Cartesian features on the unit sphere.
2. A Psiformer backbone (self-attention layers) processes the features.
3. Monopole harmonic orbitals project the backbone output into an orbital matrix.
4. A spherical Jastrow factor captures pairwise correlations.
5. The complex log-determinant gives `log psi`.

Energy estimators include spherical kinetic energy (covariant Laplacian on the sphere) and Coulomb potential energy (chord distance between electrons). The neural network naturally includes contributions from all Landau levels, going beyond lowest-Landau-level (LLL) exact diagonalization. For the derivations behind these estimators, see <project:../../guide/estimators/index.md>.

## Interpreting Energy Output

The reported `total_energy` is complex-valued — the real part is the electronic variational energy $E_v$, and the imaginary component is a finite-sampling artifact whose expectation value vanishes. Comparing $E_v$ with literature values requires post-processing corrections for background charge and finite-size effects. See <project:energy-corrections.md> for the formulas.

## Composite Fermions

The MHPO wavefunction supports composite fermion (CF) mean-field theory. Setting `wf.flux_per_elec` attaches flux quanta to each electron, reducing the effective monopole strength for the orbitals:

```bash
# Composite fermion with 2 flux quanta per electron
jaqmc hall train system.flux=10 system.nspins='[4,0]' wf.flux_per_elec=2
```

## Angular Momentum Penalties

To target states with specific angular momentum quantum numbers, use the penalty method:

```bash
# Target Lz = 0 with penalty strength 10
jaqmc hall train system.lz_penalty=10 system.lz_center=0

# Also penalize total L^2
jaqmc hall train system.lz_penalty=10 system.l2_penalty=5
```

When penalties are active, the optimizer minimizes a penalized loss instead of the bare total energy. The console output includes `Lz` and `L_square` columns to monitor convergence.

```{tip}
We recommend first converging the training **without** penalties ($\beta = 0$), then turning on penalties to select a specific angular momentum sector. We find this two-stage approach produces more stable results than training with penalties from the start.
```

## Recommended Hyperparameters

The workflow preset defaults to 200,000 training iterations so that `jaqmc hall train`
starts from a production-scale run length. If that budget fits your target state and
hardware, you can usually keep the defaults. In practice, though, some states converge
earlier while others need a longer run.

When you do tune a run, start with these hyperparameters:

The main knob is the optimization budget: choose
{cfgkey}`train.run.iterations <systems-hall-train-cfg-train-run-iterations>` based on how
long the target state takes to converge. The paper-derived settings below are a better
starting point for Hall systems than a generic rule of thumb.

For walkers, {cfgkey}`workflow.batch_size <systems-hall-train-cfg-workflow-batch-size>` controls the
variance of each VMC step. The default of 4,096 is usually a good production starting
point; increase it only if the step-to-step statistics are too noisy, and lower it for
quick tests. See <project:../../guide/sampling.md> for walker count, mixing, and burn-in
behavior.

The sampler defaults are usually reasonable. Reach for
{cfgkey}`train.sampler.steps <systems-hall-train-cfg-train-sampler-mcmc-steps>` or
{cfgkey}`train.run.burn_in <systems-hall-train-cfg-train-run-burn-in>` only when the walkers are not
mixing well or `pmove` looks unhealthy. For optimizer choice, the production default is
[train.optim.module](#hall-train-optim); use the
<project:../../guide/optimizers/index.md> guide if you want to compare it with Adam.
For Hall-specific wavefunction settings under [wf.*](#hall-train-wf), including MHPO,
Laughlin, and free states, use the training configuration reference.

For authoritative key definitions and effective defaults, see the [training configuration
reference](train.md) and use `--dry-run workflow.config.verbose=true` to inspect
the fully resolved config for your run. For checkpointing and resuming longer
jobs, see <project:../../guide/running-workflows.md>.

We used the following hyperparameters in our [paper](https://arxiv.org/abs/2412.14795) and recommend them as a starting point:

| Parameter | Value |
|-----------|-------|
| Determinants (`wf.ndets`) | 1 |
| Network layers (`wf.num_layers`) | 4 |
| Attention heads (`wf.num_heads`) | 4 |
| Attention dimension (`wf.heads_dim`) | 64 |
| Training iterations | 30,000–100,000 |

For quasiparticle/quasihole studies with the penalty method, we used an additional 20,000–40,000 iterations with penalty strengths $\beta$ in the range 0.01–0.02.

## Evaluation

After training, run evaluation to compute observables without parameter updates:

```bash
jaqmc hall train workflow.save_path=./runs/hall-train train.run.iterations=10000
jaqmc hall evaluate workflow.save_path=./runs/hall-eval \
  workflow.source_path=./runs/hall-train
```

### Additional Evaluation Estimators

The hall app includes estimators for observables beyond energy. They are disabled by default and can be enabled via config flags:

- {class}`~jaqmc.estimator.density.SphericalDensity` — Electron density as a function of polar angle $\theta$. Accumulates a histogram over evaluation steps.
- Pair correlation — Pair correlation function $g(\theta)$ from geodesic pair angles, weighted by $1/\sin\theta$. Divide the accumulated state by the step count to get the final $g(\theta)$.
- One-body reduced density matrix — One-body reduced density matrix in the monopole harmonic basis. The trace gives the number of electrons on the lowest Landau level, $N_\text{LLL}$.

Enable them via CLI or YAML:

```bash
jaqmc hall evaluate estimators.enabled.density=true estimators.enabled.pair_correlation=true
```

```yaml
estimators:
  enabled:
    density: true
    pair_correlation: true
  density:
    bins_theta: 100  # override default 50
```

## Workflow Notes

For the shared workflow patterns for debug runs, production runs, resuming,
evaluation, and reporting, see <project:../../guide/running-workflows.md>.

For hall workflows, the main system-specific choices are usually the target state and
any penalty terms, since those determine how you should choose the production training
budget and how to interpret the final run. Evaluation also commonly adds observable
estimators beyond energy, so it is worth being explicit about those when setting up or
describing a run. When reporting results, record the real part of `total_energy`, any
post-processing corrections from <project:energy-corrections.md>, and
whether angular-momentum penalties were used.

## Further Reading

- **Energy corrections** — <project:energy-corrections.md>
- **Configuration reference** — <project:train.md>, <project:eval.md>,
  and their workflow defaults
- **Estimator physics** — <project:../../guide/estimators/index.md> (includes spherical kinetic energy derivation)
- **Running evaluations** — [Workflows](#recipe-resume-evaluate)

```{toctree}
:hidden:

energy-corrections.md
train.md
eval.md
```
