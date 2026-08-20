# Molecule Reference Orbitals

Pretraining matches the neural wavefunction to the mean-field orbitals —
Hartree-Fock by default — stored in `reference.npz`. This page describes
reference resolution and standalone preparation.

Molecule evaluation never loads a reference. In training, setting
`pretrain.run.iterations=0` skips both pretraining and reference loading or
generation.

## Reference resolution

Training accepts a prepared reference file through the `reference` config key.
Without that key, JaQMC resolves the reference in this order:

1. An existing `reference.npz` in the `workflow.restore_path` run directory
   (or that path's parent if it names a checkpoint file).
2. `workflow.save_path/reference.npz`. If `workflow.restore_path` is unset,
   resolution begins here.
3. If neither file exists, process 0 runs a default UHF PySCF job in a
   temporary directory and writes `workflow.save_path/reference.npz`. The
   automatic basis is `cc-pVDZ` for all-electron elements and `ccecpccpvdz`
   for `ccecp` and PH elements; other ECP families have no automatic basis and
   need a prepared reference. Automatic generation also requires each atom to
   use the effective charge derived from `system.pp`; custom per-atom charges
   need a prepared reference.

An explicit `reference=` path skips this search and is loaded directly;
training does not search the job directory written by
`jaqmc molecule reference prepare`.

On multiple hosts, the reference path must be readable by every process. The
[shared-storage requirements](../../guide/multi-device.md#shared-storage)
apply to this file.

A training dry run without an explicit `reference` only resolves
configuration and does not search or generate an artifact. An explicit
`reference` path is still loaded and validated.

## Custom reference preparation

A custom reference supports a non-default
[basis](#basis-sets-and-pseudopotentials) or method, an ECP family without an
automatic basis, and solver input that can be inspected before execution.
`jaqmc molecule reference prepare` takes the same `system.*` definition as
training, passed with `--yml` files or as CLI overrides; keys it does not
consume, such as `workflow.*` or `pretrain.*`, are rejected. The command writes
a standalone PySCF job (the solver input only, no QMC) into `--output`; the
generated directory contains `input.py`.

### Automatic solver execution

The `--run` option runs the generated PySCF job immediately. JaQMC executes
`input.py` with the current Python
interpreter in the job directory and converts the PySCF checkpoint to
`reference.npz` in the same directory. The `--reference-output` option selects
a different filename or path.

```bash
jaqmc molecule reference prepare --yml water.yml --output ./runs/water/hf --run
jaqmc molecule train --yml water.yml reference=./runs/water/hf/reference.npz \
  workflow.save_path=./runs/water
```

### Manual solver execution

Without `--run`, the command writes the generated PySCF input without
executing it. The checkpoint can be converted after the calculation finishes:

```bash
jaqmc molecule reference prepare --yml water.yml --output ./runs/water/hf
# Optional: edit ./runs/water/hf/input.py
(cd ./runs/water/hf && python input.py)
jaqmc molecule reference convert \
  --input ./runs/water/hf/pyscf.chk \
  --output ./runs/water/hf/reference.npz
```

Training accepts the result through
`reference=./runs/water/hf/reference.npz`, as in the `--run` example above. A
launcher or scheduler command can replace the example PySCF command. Execution
from the job directory keeps relative solver outputs there.

The `--dry-run` option exits before writing the job directory.

## Solver options

`solver.*` configures the generated PySCF job. Only the `reference` commands
consume these keys.

```{eval-rst}
.. config-defaults:: jaqmc.app.molecule.config.base.MoleculeSolverConfig
   :prefix: solver
   :scope: molecule-reference
   :exclude-fields: extra
```

### Basis sets and pseudopotentials

On `water.yml`, `solver.basis.H=sto-3g` overrides hydrogen only; oxygen
keeps the automatic `cc-pVDZ` basis:

```bash
jaqmc molecule reference prepare --yml water.yml \
  --output ./runs/water/hf --run \
  solver.basis.H=sto-3g
```

A string `solver.basis` applies one PySCF basis name to every element:

```bash
jaqmc molecule reference prepare --yml water.yml \
  --output ./runs/water/hf --run \
  solver.basis=sto-3g
```

Omitted `solver.basis` uses the automatic double-zeta mapping:
`cc-pVDZ` for all-electron atoms and `ccecpccpvdz` for `ccecp` and PH
(`ccecpccpvdz` is PySCF's spelling of the ccECP-cc-pVDZ family). On a
lithium atom with `system.pp=ccecp`, that mapping is `ccecpccpvdz`:

```bash
jaqmc molecule reference prepare --output ./runs/li/hf --run \
  system.module=atom system.symbol=Li system.pp=ccecp
```

When `solver.pp` is omitted, the job uses `system.pp`. `solver.pp`
selects a different PySCF pseudopotential for the reference calculation
without changing the JaQMC system — for example, a GTH pseudopotential
for a DFT-based reference. Each override must keep the valence-electron
count from `system.pp`:

```bash
jaqmc molecule reference prepare --output ./runs/f/dft --run \
  system.module=atom system.symbol=F system.pp=ccecp \
  solver.method=UKS \
  solver.pp.F=gth-pbe-q7 \
  solver.basis=gth-dzv
```

`gth-pbe-q7` has the same seven-electron valence as fluorine with
`system.pp=ccecp`.

### SCF convergence settings

Additional `solver.*` keys, typically convergence settings such as `conv_tol`,
`max_cycle`, and `diis_space`, are forwarded to the generated PySCF mean-field
object:

```bash
jaqmc molecule reference prepare --yml water.yml \
  --output ./runs/water/hf --run \
  solver.method=RHF \
  solver.conv_tol=1.0e-10 \
  solver.max_cycle=200 \
  solver.diis_space=12
```

If a forwarded key is not an attribute of the selected PySCF mean-field object,
the generated job raises before calling `mf.kernel()`.
