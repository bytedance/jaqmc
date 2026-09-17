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

The default train preset is production-oriented. For a short sanity-check
run, see {ref}`recipe-fast-debug-run`.

The sphere radius defaults to `sqrt(Q)` where `Q = flux / 2`. Override it with
`system.radius`. For the contextual defaults used by training, see the
[training configuration reference](#hall-train-system).

## How It Works

Each electron is sampled on the sphere as `(theta, phi)`. The default
trainable ansatz is **MHPO**: a Psiformer backbone on the unit-sphere
Cartesian point, monopole-harmonic orbitals from the spinor `(u, v)`,
and a spherical Jastrow. Parameter-free `laughlin` and `free` benchmarks
share that spinor interface; see [Evaluation](#evaluation).

Energy estimators include spherical kinetic energy and Coulomb potential
energy (chord distance between electrons). Training reports `Lz` and
`L_square` on the console. For the kinetic formulation, see
[Kinetic energy](../../guide/estimators/kinetic.md#spherical-kinetic-energy);
for angular momentum, see
[Angular momentum](../../guide/estimators/angular-momentum.md);
for the full estimator guide, see
<project:../../guide/estimators/index.md>.

## Interpreting Energy Output

The reported `total_energy` is complex-valued — the real part is the electronic variational energy $E_v$, and the imaginary component is a finite-sampling artifact whose expectation value vanishes. Hall training prints that real part as `energy` (`total_energy_real`). Comparing $E_v$ with literature values requires post-processing corrections for background charge and finite-size effects. See <project:energy-corrections.md> for the formulas.

## Composite Fermions

The MHPO wavefunction supports composite fermion (CF) mean-field theory. Setting `wf.flux_per_elec` attaches flux quanta to each electron, reducing the effective monopole strength for the orbitals:

```bash
# Composite fermion with 2 flux quanta per electron
jaqmc hall train system.flux=10 system.nspins='[4,0]' wf.flux_per_elec=2
```

## Angular Momentum Penalties

The $L_z$ penalty targets `system.lz_center`. The $L^2$ penalty
drives $L^2$ toward zero. The optimizer then minimizes

$$
E + \lambda_{L_z}(L_z - L_{z,0})^2 + \lambda_{L^2} L^2
$$

instead of the bare total energy. The strengths $\lambda_{L_z}$ and
$\lambda_{L^2}$ are `system.lz_penalty` and `system.l2_penalty`. Typical
values, matching the [paper](https://arxiv.org/abs/2412.14795), are
$0.01$–$0.02$:

```bash
# Target Lz = 0
jaqmc hall train system.lz_penalty=0.02 system.lz_center=0

# Also drive L^2 toward 0
jaqmc hall train system.lz_penalty=0.02 system.l2_penalty=0.02
```

Watch `Lz` against `system.lz_center` and `L_square` toward zero. The
console `energy` column is the unpenalized variational energy
(`total_energy_real`); the optimizer minimizes `penalized_loss`. See
[Angular momentum](../../guide/estimators/angular-momentum.md).

```{tip}
Converge first **without** penalties, then resume with penalties on to
select a sector. That two-stage run is more stable than training with
penalties from the start. See {ref}`recipe-resume-evaluate`.
```

## Recommended Hyperparameters

The workflow preset uses 200,000 training iterations and MHPO
`wf.num_layers=2`. For walkers, {cfgkey}`workflow.batch_size <systems-hall-train-cfg-workflow-batch-size>` controls the
variance of each VMC step. The default of 4,096 is usually a good production starting
point; increase it only if the step-to-step statistics are too noisy, and lower it for
quick tests. See <project:../../guide/sampling.md> for walker count, mixing, and burn-in
behavior.

The sampler defaults are usually reasonable. Reach for
{cfgkey}`sampler.steps <systems-hall-train-cfg-sampler-steps>` or
{cfgkey}`train.run.burn_in <systems-hall-train-cfg-train-run-burn-in>` only when the walkers are not
mixing well or `pmove` looks unhealthy. For optimizer choice, the production default is
[train.optim.module](#hall-train-optim); use the
<project:../../guide/optimizers/index.md> guide if you want to compare it with Adam.
For Hall-specific wavefunction settings under [wf.*](#hall-train-wf),
including MHPO, use the training configuration reference. For the
analytic `laughlin` and `free` benchmarks, use the
[evaluation](eval.md) reference.

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

For quasiparticle/quasihole studies with the penalty method, we used an additional 20,000–40,000 iterations with `system.lz_penalty` and `system.l2_penalty` in the range 0.01–0.02.

## Evaluation

After training, run evaluation to compute observables without parameter updates:

```bash
jaqmc hall train workflow.save_path=./runs/hall-train train.run.iterations=10000
jaqmc hall evaluate workflow.save_path=./runs/hall-eval \
  workflow.source_path=./runs/hall-train
```

The built-in `laughlin` and `free` wavefunctions are fixed analytic ansätze with
no learnable parameters. Evaluate them directly with `jaqmc hall evaluate` — no
training checkpoint is required:

```bash
jaqmc hall evaluate workflow.save_path=./runs/laughlin-n3 \
  system.flux=6 system.nspins='[3,0]' \
  wf.module=laughlin run.iterations=100000
```

See <project:eval.md> for all evaluation options. `jaqmc hall train` does not
support `laughlin` or `free`.

### Additional Evaluation Estimators

Density, pair correlation, and the one-body reduced density matrix are
optional evaluation estimators. Enable them via config; see
[evaluation estimators](#hall-estimators) for keys and post-processing.

```bash
jaqmc hall evaluate estimators.enabled.density=true \
  estimators.enabled.pair_correlation=true estimators.enabled.one_rdm=true
```

## Reporting results

Record the real part of `total_energy`, any post-processing corrections
from <project:energy-corrections.md>, and whether angular-momentum
penalties were used. Shared debug, resume, and evaluation recipes are in
<project:../../guide/running-workflows.md>.

## Further Reading

- **Energy corrections** — <project:energy-corrections.md>
- **Configuration reference** — <project:train.md>, <project:eval.md>,
  and their workflow defaults
- **Estimator physics** — <project:../../guide/estimators/index.md>
- **Running evaluations** — {ref}`recipe-resume-evaluate`

```{toctree}
:hidden:

energy-corrections.md
Training <train.md>
Evaluation <eval.md>
```
