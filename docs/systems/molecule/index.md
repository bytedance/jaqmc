# Molecules

The `jaqmc molecule` command runs atom and molecule simulations with open
boundary conditions. Most runs start from a YAML definition and a
single `jaqmc molecule train` command. JaQMC then follows the standard
molecular workflow:

1. **Hartree-Fock (HF)** computes a reference electronic-structure solution with
   PySCF.
2. **Pretraining** matches the neural wavefunction to those orbitals.
3. **VMC training** performs the main energy optimization.

Energies are reported in Hartree (1 Ha = 27.211 eV = 627.5 kcal/mol).

## Helpful Background

- <project:../../getting-started/concepts.md> explains the main JaQMC terms used on
  this page, including VMC, walkers, and evaluation.
- <project:../../guide/running-workflows.md> covers the shared workflow mechanics:
  output directories, checkpoints, resuming, branching, and evaluation runs.

## Define the system and train

In the most general form, you define the molecule directly in YAML using the
schema shown in the [system configuration reference](#molecule-train-system).
For most runs, put that definition in a file and pass it with `--yml`.

For example, to train a neural wavefunction for water, save the following as
`water.yml`:

```yaml
system:
  atoms:
    - symbol: O
      coords: [0.0, 0.0, 0.2217]
    - symbol: H
      coords: [0.0, 1.4309, -0.8867]
    - symbol: H
      coords: [0.0, -1.4309, -0.8867]
  electron_spins: [5, 5]  # [n_up, n_down]
```

`electron_spins` gives `[n_up, n_down]` for the electrons included in the QMC
simulation. This water example is all-electron, so `[5, 5]` includes all ten
electrons. If you later add an ECP, leave out the core electrons it replaces;
`electron_spins` should count only the valence electrons JaQMC samples
explicitly.

Then run training:

```bash
jaqmc molecule train --yml water.yml workflow.save_path=./runs/water \
  pretrain.run.iterations=5000 train.run.iterations=200000
```

Those iteration counts are production-oriented. For a quick local test, use much
smaller values, such as 100 pretraining steps and 100 training steps.

`workflow.save_path` controls where JaQMC writes checkpoints and statistics such
as `train_stats.csv`. CLI overrides take precedence over YAML values, so a
common pattern is to keep the system definition in the file and tune run
settings from the command line. Use `--dry-run` to inspect the fully resolved
config without starting the job.

## Evaluate a trained model

After training finishes, run evaluation to freeze the parameters and collect
enough samples for a final energy estimate:

```bash
jaqmc molecule evaluate --yml water.yml workflow.save_path=./runs/water-eval \
  workflow.source_path=./runs/water run.iterations=2000
```

## System definition shortcuts

Direct YAML definitions are the most flexible option, but they can be verbose for
simple systems. For example, a single-atom run does not need an explicit
`atoms:` list with coordinates at the origin, and diatomic studies often vary
only the bond length. For these common cases, JaQMC provides shortcut modules
that generate the underlying configuration for you.

### Single Atoms

For a single atom, `system.module=atom` is a shortcut. You provide the element
symbol, and JaQMC fills in the matching electron spin configuration
automatically. By default it uses the all-electron count; when `system.ecp` is
set, it uses the valence count instead.

```yaml
system:
  module: atom
  symbol: Li         # Element symbol (H, He, Li, Be, ...)
  # ecp: ccecp       # Optional: effective core potential
```

Save as `atom_li.yml`, then run:

```bash
jaqmc molecule train --yml atom_li.yml workflow.save_path=./runs/atom_li
```

### Diatomic Molecules

For common two-atom systems, `system.module=diatomic` is a shortcut. You provide
the chemical formula, bond length, and optional spin for the simulated
electrons. JaQMC places the atoms along the z-axis and computes
`electron_spins` for you.

```yaml
system:
  module: diatomic
  formula: LiH        # Chemical formula (H2, LiH, N2, ClF, ...)
  bond_length: 3.015  # Distance between atoms
  unit: bohr          # Length unit for bond_length
  spin: 0             # n_up - n_down for electrons being simulated
```

Save as `li_h_diatomic.yml`, then run:

```bash
jaqmc molecule train --yml li_h_diatomic.yml workflow.save_path=./runs/li_h_diatomic
```

(molecule-ecps)=
## Effective core potentials

Most examples above are all-electron calculations: JaQMC represents every
electron in the molecule explicitly. For heavier elements, you may instead
replace core electrons with an effective core potential (ECP). The core
electrons no longer appear as QMC electrons; their effect enters through the
pseudopotential, while JaQMC samples the remaining valence electrons.

Enable an ECP by setting `system.ecp`:

```yaml
system:
  ecp: ccecp
```

Use an ECP designed for correlated many-body calculations rather than a
DFT-only pseudopotential. The correlation-consistent ECP family, `ccecp`, is the
usual choice for QMC runs.

Once an ECP is enabled, `electron_spins` describes the electrons being sampled,
not the full electron count of the physical atoms. The `atom` and `diatomic`
shortcuts use `system.ecp` to choose the valence count automatically. If you
define `atoms` and `electron_spins` directly, set `electron_spins` to the
valence-electron system you want to simulate.

For mixed systems, apply ECPs only to the elements that need them:

```yaml
system:
  ecp:
    Fe: ccecp
```

(molecule-pretrain-reference)=
## Pretrain reference settings

`pretrain.reference.*` configures the PySCF Hartree-Fock calculation used to
generate the target orbitals for pretraining. In most runs, the basis is the
only reference setting you need to choose. The default is cc-pVDZ, and you can
change it with:

```yaml
pretrain:
  reference:
    basis: sto-3g
```

If the system uses an ECP, choose a pretrain basis that matches that
pseudopotential. For example, with ccECP use the corresponding ccECP basis
family:

```yaml
system:
  module: atom
  symbol: Fe
  ecp: ccecp
pretrain:
  reference:
    basis: ccecpccpvdz
```

For mixed systems, keep the same per-element split between the physical system
and the HF reference: put ECPs in `system.ecp`, and put matching PySCF basis
choices in `pretrain.reference.basis`.

```yaml
system:
  ecp:
    Fe: ccecp
pretrain:
  reference:
    basis:
      Fe: ccecpccpvdz
      O: cc-pvdz
```

When the HF calculation itself needs tuning, use the `pretrain.reference.*`
block for PySCF solver settings. JaQMC supports
`pretrain.reference.method` (`UHF` or `RHF`) and forwards additional keys to the
selected PySCF mean-field object.

```yaml
pretrain:
  reference:
    method: RHF
    basis: cc-pvdz
    conv_tol: 1.0e-10
    max_cycle: 200
    diis_space: 12
```

Use these extra keys for SCF convergence and solver behavior tuning, such as
`conv_tol`, `max_cycle`, and related PySCF options. If a key is not supported by
the selected PySCF object, JaQMC ignores it and logs a warning.

For authoritative key definitions and defaults under `pretrain.reference.*`, see
<project:train.md>.

## Estimators

The training stage computes energy from several components: kinetic energy,
electron-nucleus potential, and, when ECPs are configured, pseudopotential
contributions. All stats keys that start with `energy:` are summed into
`total_energy` automatically.

For the full list of molecule estimators beyond energy, see the
[estimator configuration reference](#molecule-estimators). For the physics
and derivations behind each estimator, see <project:../../guide/estimators/index.md>.

## Production Settings

The workflow presets default to 2,000 pretraining iterations and 200,000
training iterations so that a bare `jaqmc molecule train ...` command is closer
to a real calculation than a smoke test. If that budget fits your system and
hardware, you can usually keep the defaults. For laptop-scale experiments, you
may want to reduce the iteration counts. See
<project:../../guide/running-workflows.md> for the shared workflow mechanics.

When you do tune a run, start with the optimization budget and walker count.

The main optimization knobs are
{cfgkey}`pretrain.run.iterations <systems-molecule-train-cfg-pretrain-run-iterations>`
and
{cfgkey}`train.run.iterations <systems-molecule-train-cfg-train-run-iterations>`.
Increase them when the energy is still drifting at the end of training; decrease
them for quick local runs.

For walkers,
{cfgkey}`workflow.batch_size <systems-molecule-train-cfg-workflow-batch-size>`
controls the variance of each VMC step, not the expressiveness of the model. In
practice, the default of 4,096 is usually enough for production runs. Lower it
for quick tests. See <project:../../guide/sampling.md> for how walker count,
acceptance rate, and burn-in interact.

For authoritative key definitions and effective defaults, see the
<project:train.md> and use
`--dry-run workflow.config.verbose=true` to inspect the fully resolved config
for your run.

:::{admonition} Checking convergence
:class: tip

Plot `total_energy` from `train_stats.csv` over training steps (see [Reading Training Statistics](../../guide/training-stats.md)). The energy should plateau. For final energy estimates, follow <project:#recipe-resume-evaluate> — training energies are biased because the parameters change at every step.
:::

:::{admonition} Multi-GPU training
:class: tip

For faster production runs on multiple GPUs, see <project:/guide/multi-device.md>.
:::

## Where To Go Next

The molecule workflow uses FermiNet and KFAC by default. To switch
architectures or optimizers:

```bash
# Use Psiformer instead of FermiNet
jaqmc molecule train wf.module=psiformer

# Use Adam instead of KFAC
jaqmc molecule train train.optim.module=optax:adam

# Reduce network size for faster experiments
jaqmc molecule train wf.hidden_dims_single='[128, 128]' wf.hidden_dims_double='[16, 16]'
```

After you can run a basic molecule workflow, these pages cover the usual next
questions:

- **Configuration reference**: <project:train.md> and <project:eval.md> list the resolved
  workflow defaults and every supported key.
- **Training diagnostics**: <project:../../guide/training-stats.md>
  shows how to interpret `train_stats.csv` and check convergence.
- **Estimator physics**: <project:../../guide/estimators/index.md>
  explains the energy terms and optional observables used in molecule runs.
- **Wavefunction choices**: <project:../../guide/wavefunction.md>
  compares FermiNet, Psiformer, and their main tuning knobs.
- **Optimizer choices**: <project:../../guide/optimizers/index.md> explains when
  to keep KFAC and when to switch to alternatives such as Adam.
- **Troubleshooting**: <project:../../guide/troubleshooting.md> covers
  common failures such as unstable optimization, NaNs, and recovery steps.
- **Periodic systems**: <project:../solid/index.md> is the matching entry point for
  crystals and other systems with periodic boundary conditions.

```{toctree}
:hidden:

train.md
eval.md
```
