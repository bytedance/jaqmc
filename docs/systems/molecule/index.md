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

Direct molecule YAML uses Bohr by default. If your source geometry is in
angstrom, set `system.unit: angstrom`. JaQMC converts the coordinates to Bohr
before Hartree-Fock and training.

For arbitrary molecules, the config describes the nuclei plus the electronic
constraints rather than separate spin counts directly. JaQMC resolves each atom's
effective charge from its element symbol and optional `system.pp`, then derives
the spin-up and spin-down electron counts from the resolved total electron
count plus `system.s_z`. Use `system.total_charge` when the simulated system is
ionic.

For example, to train a neural wavefunction for water from an angstrom-scale
geometry, save the following as `water.yml`:

```yaml
system:
  unit: angstrom
  atoms:
    - symbol: O
      coords: [0.0, 0.0, 0.0]
    - symbol: H
      coords: [0.0, 0.757, 0.586]
    - symbol: H
      coords: [0.0, -0.757, 0.586]
  s_z: 0                  # singlet
```

This water example is neutral and all-electron, so JaQMC resolves ten explicit
electrons and assigns five spin-up and five spin-down electrons from `s_z: 0`.
If you later enable an ECP or PH pseudopotential, JaQMC derives the
valence-electron count automatically after pseudopotential resolution. For
ions, add `system.total_charge`; for example, `total_charge: 1` removes one
explicit electron from the simulated system.

If you need finer control over charge resolution or electron initialization:

- `atoms[*].charge` overrides the effective charge seen by the simulated electrons.
- `atoms[*].initialization.local_s_z` biases the initial alpha/beta split near one atom.
- `atoms[*].initialization.local_charge` shifts how many electrons are seeded near one atom initially.

Those per-atom initialization fields affect only the starting walker positions,
not the physical Hamiltonian.

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
symbol, and JaQMC places it at the origin and fills in the default `s_z` for
the neutral atom automatically. By default it uses the all-electron charge;
when `system.pp` selects a pseudopotential for that atom, the derived explicit
electron count switches to the corresponding valence count automatically.

```yaml
system:
  module: atom
  symbol: Li         # Element symbol (H, He, Li, Be, ...)
  # pp: ccecp        # Optional: pseudopotential ("ph" or an ECP name)
```

Save as `atom_li.yml`, then run:

```bash
jaqmc molecule train --yml atom_li.yml workflow.save_path=./runs/atom_li
```

### Diatomic Molecules

For common two-atom systems, `system.module=diatomic` is a shortcut. You provide
the chemical formula, bond length, and optional `s_z` for the simulated
electrons. JaQMC places the atoms along the z-axis, resolves atom charges, and
derives the resulting spin-up and spin-down counts for you.

```yaml
system:
  module: diatomic
  formula: LiH        # Chemical formula (H2, LiH, N2, ClF, ...)
  bond_length: 3.015  # Distance between atoms
  unit: bohr          # Length unit for bond_length
  s_z: 0              # singlet
```

Save as `li_h_diatomic.yml`, then run:

```bash
jaqmc molecule train --yml li_h_diatomic.yml workflow.save_path=./runs/li_h_diatomic
```

(molecule-pseudopotentials)=
## Pseudopotentials

Most examples above are all-electron calculations: JaQMC represents every
electron in the molecule explicitly. For heavier elements, you may instead
replace core electrons with an effective core potential (ECP). The core
electrons no longer appear as QMC electrons; their effect enters through the
pseudopotential, while JaQMC samples the remaining valence electrons.

Pseudopotentials are configured through the unified `system.pp` field. A string
applies one pseudopotential selector to every atom; a mapping selects per
element. Two pseudopotential families are supported:

- An ECP name (for example, `ccecp`) selects a semi-local effective core
  potential resolved by PySCF. See <project:/guide/estimators/ecp.md>.
- The reserved literal `"ph"` selects the local Pseudo-Hamiltonian family,
  parallel to the semi-local ECP family. See
  <project:/guide/estimators/ph.md>.

To use an ECP, set `system.pp` to an ECP name or mapping:

```yaml
system:
  pp: ccecp
```

Use an ECP designed for correlated many-body calculations rather than a
DFT-only pseudopotential. The correlation-consistent ECP family, `ccecp`, is the
usual choice for QMC runs.

Atoms whose element is not in the `pp` mapping are treated all-electron, so a
single system may mix PH, semi-local ECP, and all-electron elements freely.

Once an ECP is enabled, JaQMC derives the explicit electron count from the
valence system rather than from the all-electron atoms. The `atom` and
`diatomic` shortcuts use `system.pp` to choose the corresponding
simulated-electron count automatically. If you define `atoms` directly, set
`system.s_z` to the desired value for the explicit electrons, and add
`system.total_charge` if the simulated valence system is charged.

For mixed systems, apply ECPs only to the elements that need them:

```yaml
system:
  pp:
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
  pp: ccecp
pretrain:
  reference:
    basis: ccecpccpvdz
```

For mixed systems, keep the same per-element split between the physical system
and the HF reference: put pseudopotential choices in `system.pp`, and put
matching PySCF basis choices in `pretrain.reference.basis`.
```yaml
system:
  pp:
    Fe: ph
    Li: ccecp
pretrain:
  reference:
    basis:
      Fe: ccecpccpvdz
      Li: ccecpccpvdz
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
electron-nucleus potential, and, when a pseudopotential is configured through
`system.pp`, pseudopotential contributions from ECP and/or PH atoms. All stats
keys that start with `energy:` are summed into `total_energy` automatically.

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

Plot `total_energy` from `train_stats.csv` over training steps (see [Reading Training Statistics](../../guide/training-stats.md)). The energy should plateau. For final energy estimates, follow [Resume, branch, or evaluate](#recipe-resume-evaluate) — training energies are biased because the parameters change at every step.
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
