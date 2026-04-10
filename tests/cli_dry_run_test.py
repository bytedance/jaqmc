# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from jaqmc.app.cli import cli


@dataclass(frozen=True)
class CliDryRunCase:
    id: str
    command: str
    files: dict[str, str] = field(default_factory=dict)


CASES = [
    CliDryRunCase("hydrogen_atom_train", "hydrogen-atom train"),
    CliDryRunCase(
        "molecule_atom_dotlist",
        "molecule train system.module=atom system.symbol=Li",
    ),
    CliDryRunCase(
        "molecule_water_yaml",
        "molecule train --yml water.yml",
        {
            "water.yml": """
system:
  atoms:
    - symbol: O
      coords: [0.0, 0.0, 0.0]
    - symbol: H
      coords: [0.0, 0.757, 0.586]
    - symbol: H
      coords: [0.0, -0.757, 0.586]
  electron_spins: [5, 5]
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
pretrain:
  run:
    iterations: 1
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "molecule_yaml_layering",
        "molecule train --yml base.yml --yml override.yml train.run.iterations=2",
        {
            "base.yml": """
system:
  module: atom
  symbol: H
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
train:
  run:
    iterations: 1
""",
            "override.yml": """
system:
  symbol: Li
""",
        },
    ),
    CliDryRunCase(
        "molecule_diatomic_yaml",
        "molecule train --yml lih_diatomic.yml",
        {
            "lih_diatomic.yml": """
system:
  module: diatomic
  formula: LiH
  bond_length: 3.015
  unit: bohr
  basis: sto-3g
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
pretrain:
  run:
    iterations: 1
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "molecule_ecp_atom_yaml",
        "molecule train --yml fe_ecp.yml",
        {
            "fe_ecp.yml": """
system:
  module: atom
  symbol: Fe
  basis: ccecpccpvdz
  ecp: ccecp
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
pretrain:
  run:
    iterations: 1
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "molecule_psiformer_dotlist",
        "molecule train wf.module=psiformer workflow.batch_size=4 "
        "wf.num_layers=1 wf.num_heads=1 wf.heads_dim=4 "
        "wf.mlp_hidden_dims='[4]' pretrain.run.iterations=1 "
        "train.run.iterations=1",
    ),
    CliDryRunCase(
        "solid_two_atom_chain_yaml",
        "solid train --yml solid.yml",
        {
            "solid.yml": """
system:
  module: two_atom_chain
  vacuum_separation: 10.0
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
pretrain:
  run:
    iterations: 1
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "solid_arbitrary_lih_yaml",
        "solid train --yml lih_solid.yml",
        {
            "lih_solid.yml": """
system:
  lattice_vectors:
    - [0.0, 3.78, 3.78]
    - [3.78, 0.0, 3.78]
    - [3.78, 3.78, 0.0]
  atoms:
    - symbol: Li
      coords: [0.0, 0.0, 0.0]
    - symbol: H
      coords: [3.78, 3.78, 3.78]
  electron_spins: [2, 2]
  basis: sto-3g
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
pretrain:
  run:
    iterations: 1
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "solid_rock_salt_yaml",
        "solid train --yml rock_salt.yml",
        {
            "rock_salt.yml": """
system:
  module: rock_salt
  symbol_a: Li
  symbol_b: H
  lattice_constant: 4.0
  unit: angstrom
  basis: sto-3g
workflow:
  batch_size: 4
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
pretrain:
  run:
    iterations: 1
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "hall_train_yaml",
        "hall train --yml hall.yml",
        {
            "hall.yml": """
system:
  nspins: [3, 0]
  flux: 6
workflow:
  batch_size: 4
train:
  run:
    iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "hall_train_penalty_dotlist",
        "hall train system.lz_penalty=10 system.lz_center=0 "
        "workflow.batch_size=4 train.run.iterations=1",
    ),
    CliDryRunCase(
        "hall_train_composite_fermion_dotlist",
        "hall train system.flux=10 system.nspins='[4,0]' "
        "wf.flux_per_elec=2 workflow.batch_size=4 train.run.iterations=1",
    ),
    CliDryRunCase(
        "molecule_evaluate_yaml",
        "molecule evaluate --yml eval.yml",
        {
            "eval.yml": """
system:
  module: atom
  symbol: H
workflow:
  batch_size: 4
  source_path: source
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
run:
  iterations: 1
""",
        },
    ),
    CliDryRunCase(
        "solid_evaluate_density_yaml",
        "solid evaluate --yml solid_eval.yml",
        {
            "solid_eval.yml": """
system:
  module: rock_salt
workflow:
  batch_size: 4
  source_path: source
wf:
  hidden_dims_single: [4, 4]
  hidden_dims_double: [2, 2]
run:
  iterations: 1
estimators:
  enabled:
    density: true
""",
        },
    ),
    CliDryRunCase(
        "hall_evaluate_observables_dotlist",
        "hall evaluate workflow.batch_size=4 workflow.source_path=source "
        "run.iterations=1 estimators.enabled.density=true "
        "estimators.enabled.pair_correlation=true "
        "estimators.enabled.one_rdm=true",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_cli_command_dry_run(
    case: CliDryRunCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name, text in case.files.items():
        (tmp_path / name).write_text(text.strip() + "\n", encoding="utf8")
    result = CliRunner().invoke(cli, [*shlex.split(case.command), "--dry-run"])
    assert result.exit_code == 0, f"command: {case.command}\noutput:\n{result.output}"


def test_cli_verbose_config_dotlist(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="jaqmc.utils.config")

    result = CliRunner().invoke(
        cli,
        [
            "molecule",
            "train",
            "--dry-run",
            "workflow.config.verbose=true",
            "workflow.batch_size=4",
            "wf.hidden_dims_single=[4,4]",
            "wf.hidden_dims_double=[2,2]",
            "pretrain.run.iterations=1",
            "train.run.iterations=1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "verbose: true" in caplog.text
    assert "Base configuration for workflows." in caplog.text
