# Solid Reference Orbitals

The orbital reference (`reference.npz`) supplies the occupied k-points for the
ansatz and, when pretraining is enabled, the target orbitals. This page covers
reference resolution by `train` and `evaluate`, plus standalone preparation
with PySCF or Quantum ESPRESSO.

## Reference resolution

Resolution follows the same rules as for
[molecule runs](../molecule/reference.md#reference-resolution). Both `train`
and `evaluate` accept a prepared file through the `reference` config key.
Without one, `train` checks `workflow.restore_path` and then
`workflow.save_path`. Evaluation checks `workflow.restore_path`,
`workflow.source_path`, and `workflow.save_path`, in that order. If none
contains `reference.npz`, the command generates a default PySCF reference in
`workflow.save_path`. Solid runs differ in three other ways:

- Evaluation loads the reference as well — the ansatz reads the occupied
  k-points from it — and additionally searches the `workflow.source_path` run
  directory.
- The automatic job is KUHF, with automatic basis `cc-pVDZ` for all-electron
  and `ccecpccpvdz` for `ccecp` elements; other ECP families need a prepared
  reference. Automatic generation also requires each atom to use the effective
  charge derived from `system.pp`; custom per-atom charges need a prepared
  reference.
- An explicit path is loaded in place and not copied into
  `workflow.save_path`; evaluation therefore requires the same explicit path.

Whether found by the search or passed explicitly, a loaded reference must use
the same unreduced twisted-supercell k-point mesh as the QMC run, modulo
primitive reciprocal lattice vectors; a mismatched reference is rejected when
loaded.

## Custom reference preparation

A custom reference supports a non-default
[basis](#basis-sets-and-pseudopotentials) or method, an ECP family without an
automatic basis, solver input that can be inspected before execution, and
plane-wave references from Quantum ESPRESSO.
`jaqmc solid reference prepare` takes the same `system.*` definition as
training, passed with `--yml` files or as CLI overrides; keys it does not
consume, such as `workflow.*` or `pretrain.*`, are rejected. The command writes
a standalone solver job (the solver input only, no QMC) into `--output`. The
selected solver is PySCF by default, or Quantum ESPRESSO when
`solver.module=qe`.

### Automatic solver execution

The `--run` option runs the generated solver job immediately. JaQMC executes
the solver in the job directory and converts its output to `reference.npz` in
the same directory. The `--reference-output` option selects a different
filename or path.

For the default PySCF solver:

```bash
jaqmc solid reference prepare --yml lih_solid.yml \
  --output ./runs/lih_solid/hf --run
jaqmc solid train --yml lih_solid.yml \
  reference=./runs/lih_solid/hf/reference.npz \
  workflow.save_path=./runs/lih_solid
jaqmc solid evaluate --yml lih_solid.yml \
  reference=./runs/lih_solid/hf/reference.npz \
  workflow.save_path=./runs/lih_solid/eval \
  workflow.source_path=./runs/lih_solid
```

The `solver.module=qe` setting selects Quantum ESPRESSO and requires Unified
Pseudopotential Format (UPF) file paths. With `--run`, JaQMC executes
`pw.x -in qe.in` in the job directory, so `pw.x` must be on `PATH`. A QE build
that needs an MPI launcher, scheduler, or a
nonstandard executable path must use the manual steps under
[Manual solver execution](#manual-solver-execution). The
[LiH example](#example-lih-qe-reference) below uses the same system YAML as
training plus a second solver file and shows the corresponding `prepare`,
`train`, and `evaluate` commands.

### Manual solver execution

Without `--run`, the command writes the generated solver input without
executing it. The solver output can be converted to `reference.npz` after the
calculation finishes.

For PySCF, the generated checkpoint is converted with:

```bash
jaqmc solid reference prepare --yml lih_solid.yml --output ./runs/lih_solid/hf
# Optional: edit ./runs/lih_solid/hf/input.py
(cd ./runs/lih_solid/hf && python input.py)
jaqmc solid reference convert \
  --input ./runs/lih_solid/hf/pyscf.chk \
  --output ./runs/lih_solid/hf/reference.npz
```

`convert` auto-detects the input format — a PySCF checkpoint file or a
Quantum ESPRESSO `{prefix}.save` directory. The `--source pyscf` and
`--source qe` options override detection.

Training accepts the result through
`reference=./runs/lih_solid/hf/reference.npz`, as in the `--run` example.

For Quantum ESPRESSO, the manual variant of the
[LiH example](#example-lih-qe-reference) omits `--run` and converts
`{prefix}.save` after `pw.x`.

A launcher or scheduler command can replace the example solver commands.
Execution from the job directory keeps relative solver outputs there.

The `--dry-run` option exits before writing the job directory.

## PySCF references

PySCF is the default solid reference solver.

### PySCF solver options

`solver.*` configures the generated periodic PySCF job. Only the `reference`
commands consume these keys.

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.pyscf.SolidPySCFSolverConfig
   :prefix: solver
   :scope: solid-pyscf-reference
   :exclude-fields: extra
```

### Basis sets and pseudopotentials

On the all-electron LiH cell in `lih_solid.yml`, `solver.basis.H=sto-3g`
overrides hydrogen only; lithium keeps the automatic `cc-pVDZ` basis:

```bash
jaqmc solid reference prepare --yml lih_solid.yml \
  --output ./runs/lih_solid/hf --run \
  solver.basis.H=sto-3g
```

A string `solver.basis` applies one PySCF basis name to every element:

```bash
jaqmc solid reference prepare --yml lih_solid.yml \
  --output ./runs/lih_solid/hf --run \
  solver.basis=sto-3g
```

Omitted `solver.basis` uses the automatic double-zeta mapping:
`cc-pVDZ` for all-electron atoms and `ccecpccpvdz` for `ccecp`
(`ccecpccpvdz` is PySCF's spelling of the ccECP-cc-pVDZ family). With
`system.pp.Li=ccecp` on that cell, the mapping is `Li=ccecpccpvdz` and
`H=cc-pVDZ`:

```bash
jaqmc solid reference prepare --yml lih_solid.yml \
  --output ./runs/lih_solid/hf --run \
  system.pp.Li=ccecp
```

When `solver.pp` is omitted, the job uses `system.pp`. `solver.pp`
selects a different PySCF pseudopotential for the reference calculation
without changing the JaQMC system — for example, a GTH pseudopotential
for a DFT-based reference. Each override must keep the valence-electron
count from `system.pp`:

```bash
jaqmc solid reference prepare --output ./runs/f_solid/dft --run \
  system.module=two_atom_chain system.symbol=F system.pp=ccecp \
  solver.method=KRKS \
  solver.pp.F=gth-pbe-q7 \
  solver.basis=gth-dzv
```

`gth-pbe-q7` has the same seven-electron valence as fluorine with
`system.pp=ccecp`.

### SCF convergence settings

Additional `solver.*` keys, typically convergence settings such as `conv_tol`
and `max_cycle`, are forwarded to the generated PySCF mean-field object:

```bash
jaqmc solid reference prepare --yml lih_solid.yml \
  --output ./runs/lih_solid/hf --run \
  solver.conv_tol=1.0e-10 \
  solver.max_cycle=200
```

If a forwarded key is not an attribute of the selected PySCF mean-field
object, the generated job raises before calling `mf.kernel()`.

## Quantum ESPRESSO references

Quantum ESPRESSO (QE) provides a plane-wave reference. JaQMC currently accepts
only collinear PBE SCF output from QE 7.x, with wavefunction files written in
HDF5 format. Output from other exchange-correlation functionals is rejected.
Conversion expects an unreduced k-point mesh (`nosym=.true.`,
`noinv=.true.`) on the QMC twisted-supercell k-points, which the generated
input already sets.

### QE prerequisites

QE reference jobs require UPFs made for the PBE functional and a `pw.x` build
with HDF5 support. For a CMake QE build, `-DQE_ENABLE_HDF5=ON` enables that
support. The [official Quantum ESPRESSO CMake build
guide](https://gitlab.com/QEF/q-e/-/wikis/Developers/CMake-build-system)
describes platform prerequisites and build details.

### UPF and charge requirements

Because `pw.x` is a plane-wave pseudopotential solver, the generated QE input
needs a UPF file for **every** element. This is a QE input requirement.

JaQMC reads the `z_valence` declared in each UPF header and raises if it does
not match the charge JaQMC simulates for that element. All-electron elements
therefore need no-core UPFs, which in practice are readily available only for
light elements such as H or He; for heavier all-electron systems the PySCF
reference path is the practical option.

`solver.pseudo_file` requires an entry for every element. JaQMC does not infer
UPF filenames from `system.pp`; each entry names a local file under
`solver.pseudo_dir`.

### Example: LiH QE reference

The following LiH configuration uses the one-electron `ccecp` valence charge
for Li and an H UPF with `z_valence="1"` for all-electron H, so both UPF
headers declare a valence charge of 1. The example assumes that the UPFs from
the sources below are stored in one local directory and that their local
filenames appear in `qe_solver.yml`. The following system file is
`lih_solid.yml` (the LiH cell from the overview page, plus the `pp` entry):

```yaml
system:
  unit: angstrom
  pp:
    Li: ccecp
  lattice:
    a: [0.0, 2.0, 2.0]
    b: [2.0, 0.0, 2.0]
    c: [2.0, 2.0, 0.0]
  atoms:
    - symbol: Li
      frac_coords: [0.0, 0.0, 0.0]
    - symbol: H
      frac_coords: [0.5, 0.5, 0.5]
  s_z: 0
```

The corresponding `qe_solver.yml` contains the QE settings. `pseudo_dir` is a
directory on the machine that runs QE. JaQMC does not download UPF files.

```yaml
solver:
  module: qe
  pseudo_dir: path/to/upf
  prefix: jaqmc
  pseudo_file:
    Li: Li.ccECP.UPF
    H: H_ONCV_PBE-1.0.upf
```

The Li ccECP UPF in this example comes from the
[ccECP pseudopotential library](https://pseudopotentiallibrary.org/) and has
the local filename `Li.ccECP.UPF`. The H file is a no-core PBE UPF such as
[`H_ONCV_PBE-1.0.upf`](http://www.quantum-simulation.org/potentials/sg15_oncv/upf/H_ONCV_PBE-1.0.upf)
from the
[SG15 ONCV library](http://www.quantum-simulation.org/potentials/sg15_oncv/).
Different local filenames require matching `solver.pseudo_file` values.

```bash
jaqmc solid reference prepare --yml lih_solid.yml --yml qe_solver.yml \
  --output ./runs/lih_solid/qe --run
jaqmc solid train --yml lih_solid.yml \
  reference=./runs/lih_solid/qe/reference.npz \
  workflow.save_path=./runs/lih_solid
jaqmc solid evaluate --yml lih_solid.yml \
  reference=./runs/lih_solid/qe/reference.npz \
  workflow.save_path=./runs/lih_solid/eval \
  workflow.source_path=./runs/lih_solid
```

The manual workflow writes `qe.in` without `--run`. After any edits, execution
of `pw.x -in qe.in` from `./runs/lih_solid/qe` produces `{prefix}.save`
(default prefix `jaqmc`) for conversion:

```bash
jaqmc solid reference convert --source qe \
  --input ./runs/lih_solid/qe/jaqmc.save \
  --output ./runs/lih_solid/qe/reference.npz
```

### Quantum ESPRESSO solver options

The QE solver is selected by setting `solver.module=qe`.

```{eval-rst}
.. config-defaults:: jaqmc.app.solid.config.qe.SolidQESolverConfig
   :prefix: solver
   :scope: solid-qe-reference
```

### Metallic QE references

For metallic systems, `solver.smearing` enables QE smearing and
`solver.degauss` specifies a positive width in Rydberg. The [QE input
documentation](https://www.quantum-espresso.org/Doc/INPUT_PW.html#smearing)
describes the available smearing functions:

```yaml
solver:
  module: qe
  smearing: mv
  degauss: 0.01
```

The value `smearing: off` selects fixed occupations. When QE reports
fractional occupations, conversion logs a warning and fills the lowest-energy orbitals.
Without spin polarization, QE counts spin-up and spin-down electrons together,
with up to two electrons per orbital, so conversion assigns half of them to
each spin. A spin-polarized calculation already counts the two spins
separately.

### Generated QE input constraints

JaQMC writes and expects an explicit, unreduced QMC
[twisted-supercell](../../guide/periodic-boundaries.md) k-point list which sets
`nosym=.true.` and `noinv=.true.`. QE k-point weights cannot
recover the plane-wave coefficients omitted from a symmetry- or
time-reversal-reduced mesh. Existing conventional QE outputs that used either
reduction cannot be converted.

A nonzero `system.s_z` selects collinear spin (`nspin = 2`) and sets
`tot_magnetization` to `n_alpha - n_beta`. The generated k-point list is the QMC
twisted-supercell mesh, not a separate DFT k-grid.
