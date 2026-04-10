# Solids

The `jaqmc solid` command simulates periodic systems (crystals) using periodic
boundary conditions. Most runs start from a YAML definition and a
single `jaqmc solid train` command. JaQMC then follows the same three-stage
workflow used for [molecules](../molecule/index.md):

1. **Hartree-Fock (HF)** computes reference orbitals with PySCF.
2. **Pretraining** matches the neural wavefunction to those orbitals.
3. **VMC training** performs the main energy optimization.

Energies are reported in Hartree (1 Ha = 27.211 eV = 627.5 kcal/mol). For
background on QMC methods for periodic systems, see [Foulkes et al., Rev. Mod.
Phys. 73, 33 (2001)](https://doi.org/10.1103/RevModPhys.73.33). The neural
network ansatz for solids follows [Li et al., Nat. Commun. 13, 7895
(2022)](https://doi.org/10.1038/s41467-022-35627-1).

## Helpful Background

- <project:../../getting-started/concepts.md> explains the main JaQMC terms used on
  this page, including VMC, walkers, and evaluation.
- <project:../../guide/running-workflows.md> covers the shared workflow mechanics:
  output directories, checkpoints, resuming, branching, and evaluation runs.

## Define the system and train

In the most general form, you define the primitive cell directly in YAML using
the schema shown in the [system configuration reference](#solid-train-system).
This is the most flexible interface and the source of
truth for solid systems.

When you define a crystal directly, enter `lattice_vectors` and atomic
coordinates in Bohr. The shortcut modules described later accept their own
`unit` field and convert internally.

For example, save the following LiH primitive cell as `lih_solid.yml`:

```yaml
system:
  lattice_vectors:            # 3x3 primitive cell vectors (Bohr)
    - [0.0, 3.78, 3.78]
    - [3.78, 0.0, 3.78]
    - [3.78, 3.78, 0.0]
  atoms:
    - symbol: Li
      coords: [0.0, 0.0, 0.0]
    - symbol: H
      coords: [3.78, 3.78, 3.78]
  electron_spins: [2, 2]     # [n_up, n_down] per primitive cell
  basis: sto-3g
```

Then run training:

```bash
jaqmc solid train --yml lih_solid.yml workflow.save_path=./runs/lih_solid \
  pretrain.run.iterations=5000 train.run.iterations=50000
```

Those iteration counts are production-oriented. For a quick local test, use much
smaller values.

`workflow.save_path` controls where JaQMC writes checkpoints and statistics such
as `train_stats.csv`. CLI overrides take precedence over YAML values, so a
common pattern is to keep the system definition in the file and tune run
settings from the command line. Use `--dry-run` to inspect the fully resolved
config without starting the job.

## System definition shortcuts

Direct YAML definitions are the most flexible option, but they can be verbose
for common crystal families. For example, a rock-salt structure is usually
identified by its species and lattice constant, and a simple chain study often
varies only the bond length. For these cases, JaQMC provides shortcut modules
that generate the underlying configuration for you.

### Rock Salt

For FCC rock-salt structures such as LiH or NaCl, `system.module=rock_salt` is
a shortcut. You provide the species and lattice constant, and JaQMC builds the
primitive cell and fills in the corresponding electron counts automatically.

```yaml
system:
  module: rock_salt
  symbol_a: Li
  symbol_b: H
  lattice_constant: 4.0     # in angstrom by default
  unit: angstrom            # or "bohr"
  # supercell: [2, 2, 2]    # Optional diagonal supercell shorthand
  basis: sto-3g
```

Save as `rock_salt.yml`, then run:

```bash
jaqmc solid train --yml rock_salt.yml workflow.save_path=./runs/rock_salt
```

### Two-Atom Chain

For simple one-dimensional test systems, `system.module=two_atom_chain` is a
shortcut. You provide the element, bond length, and optional spin; JaQMC builds
a primitive cell with two atoms along the chain direction.

```yaml
system:
  module: two_atom_chain
  symbol: H                  # Atomic symbol (both atoms are the same element)
  bond_length: 1.8           # Distance between atoms along the chain
  unit: bohr                 # or "angstrom"
  spin: 0                    # n_up - n_down per primitive cell
  # supercell: 4             # Optional repetition along the chain direction
  basis: sto-3g
```

Save as `two_atom_chain.yml`, then run:

```bash
jaqmc solid train --yml two_atom_chain.yml workflow.save_path=./runs/two_atom_chain
```

Basis sets and ECPs work the same as for
[molecules](#molecule-basis-sets-and-ecps).

## Supercell Expansion

The `supercell_matrix` field expands the primitive cell into a larger
simulation cell: $\mathbf{A}_\text{super} = S \cdot \mathbf{A}_\text{prim}$,
where $S$ is a $3 \times 3$ integer matrix. The number of primitive cells in
the supercell is $\det(S)$, and all quantities such as electrons, atoms, and
k-points scale accordingly.

For example, adding `supercell_matrix: [[2,0,0],[0,2,0],[0,0,2]]` to the LiH
crystal above creates a $2 \times 2 \times 2$ supercell with 8 primitive cells
and 8 times the electrons. Diagonal matrices expand along each lattice
direction independently; non-diagonal matrices allow more general
transformations, such as converting an FCC primitive cell to a conventional
cubic cell.

Larger supercells reduce finite-size errors but increase computational cost,
because wavefunction evaluation scales cubically with electron count. For
production runs, start with the primitive cell and increase the supercell size
until the energy per electron converges. [Twist averaging](#twisted-boundary-conditions)
is another technique for reducing finite-size
errors without increasing the supercell.

The shortcut modules (`rock_salt`, `two_atom_chain`) accept a simplified
`supercell` shorthand, as shown in the YAML examples above.

## Evaluate a trained model

After training finishes, run evaluation to freeze the parameters and collect
samples for the final observables:

```bash
jaqmc solid evaluate --yml lih_solid.yml workflow.save_path=./runs/lih_solid-eval \
  workflow.source_path=./runs/lih_solid
```

To run multiple evaluations with different settings, use a different
`save_path` for each.

:::{note}
The total energy in solid simulations is complex-valued because the wavefunction
uses complex [Bloch phases](#bloch-phases-in-the-wavefunction). The
reported `total_energy` is the real part; the imaginary component is a
finite-sampling artifact whose expectation value vanishes.
:::

## Production Settings

The workflow presets default to 2,000 pretraining iterations and 200,000
training iterations so that a bare `jaqmc solid train ...` command is usable
for a real run. If that budget fits your cell size and hardware, you can
usually keep the defaults. Primitive cells and toy systems may converge
earlier, while larger supercells may need more steps.
See <project:../../guide/running-workflows.md> for the shared workflow mechanics.

When you do tune a run, start with the optimization budget, walker count, and
supercell size.

The main optimization knobs are
{cfgkey}`pretrain.run.iterations <systems-solid-train-cfg-pretrain-run-iterations>` and
{cfgkey}`train.run.iterations <systems-solid-train-cfg-train-run-iterations>` based on how
long the energy takes to settle. The table below gives solid-specific starting
points for primitive cells and larger supercells.

For walkers,
{cfgkey}`workflow.batch_size <systems-solid-train-cfg-workflow-batch-size>`
controls the variance of each VMC step, not the system size itself. In
practice, the default of 4,096 is usually enough for production runs, and it is
a good place to start even for larger cells. Do not increase it just because you are using more
GPUs. See <project:../../guide/sampling.md> for walker count and MCMC tuning,
and [Multi-Device](../../guide/multi-device.md) for how walkers are distributed
across GPUs.

For authoritative key definitions and effective defaults, see the [training configuration
<project:train.md> and use `--dry-run workflow.config.verbose=true` to inspect
the fully resolved config for your run. For checkpointing and resuming longer
jobs, see <project:../../guide/running-workflows.md>.

:::{admonition} Checking convergence
:class: tip

Plot `total_energy` from `train_stats.csv` over training steps. For solids,
convergence is typically slower and noisier than for molecules. For final
energy estimates, always run an evaluation, because training energies are
biased while the parameters are still changing.
:::

:::{admonition} Multi-GPU training
:class: tip

Solid simulations benefit significantly from multi-GPU parallelism. See <project:/guide/multi-device.md> for setup instructions.
:::

## Where To Go Next

- **Periodic boundary conditions**: [Distance functions, Bloch phases, and twisted boundary conditions](../../guide/periodic-boundaries.md)
  explains the solid-specific concepts behind PBC runs.
- **Configuration reference**: <project:train.md> and <project:eval.md> list the resolved
  workflow defaults and every supported key.
- **Training diagnostics**: <project:../../guide/training-stats.md>
  shows how to interpret `train_stats.csv` and judge noisy solid convergence.
- **Estimator physics**: <project:../../guide/estimators/index.md>
  covers periodic Coulomb terms such as Ewald summation.
- **Optimizer choices**: <project:../../guide/optimizers/index.md> explains the
  available training optimizers and their tradeoffs.
- **Troubleshooting**: <project:../../guide/troubleshooting.md> covers
  common failures such as unstable optimization, NaNs, and recovery steps.
- **Open-boundary systems**: <project:../molecule/index.md> is the matching entry
  point for atoms and molecules without periodic boundary conditions.

```{toctree}
:hidden:

train.md
eval.md
```
