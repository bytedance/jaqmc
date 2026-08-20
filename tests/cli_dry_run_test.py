# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from jaqmc.app.cli import cli


@dataclass(frozen=True)
class CliDryRunCase:
    id: str
    app: str
    action: str
    dotlist: tuple[str, ...] = ()
    files: dict[str, str] = field(default_factory=dict)
    yaml_files: tuple[str, ...] = ()

    @property
    def argv(self) -> list[str]:
        argv = [self.app, self.action]
        for path in self.yaml_files:
            argv.extend(("--yml", path))
        return [*argv, *self.dotlist]


CASES = [
    CliDryRunCase("hydrogen_atom_train", "hydrogen-atom", "train"),
    CliDryRunCase(
        "molecule_atom_dotlist",
        "molecule",
        "train",
        ("system.module=atom", "system.symbol=Li"),
    ),
    CliDryRunCase(
        "molecule_water_angstrom_yaml",
        "molecule",
        "train",
        files={
            "water_angstrom.yml": """
system:
  unit: angstrom
  atoms:
    - symbol: O
      coords: [0.0, 0.0, 0.0]
    - symbol: H
      coords: [0.0, 0.757, 0.586]
    - symbol: H
      coords: [0.0, -0.757, 0.586]
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
        yaml_files=("water_angstrom.yml",),
    ),
    CliDryRunCase(
        "molecule_water_yaml",
        "molecule",
        "train",
        files={
            "water.yml": """
system:
  atoms:
    - symbol: O
      coords: [0.0, 0.0, 0.0]
    - symbol: H
      coords: [0.0, 0.757, 0.586]
    - symbol: H
      coords: [0.0, -0.757, 0.586]
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
        yaml_files=("water.yml",),
    ),
    CliDryRunCase(
        "molecule_yaml_layering",
        "molecule",
        "train",
        dotlist=("train.run.iterations=2",),
        files={
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
        yaml_files=("base.yml", "override.yml"),
    ),
    CliDryRunCase(
        "molecule_diatomic_yaml",
        "molecule",
        "train",
        files={
            "lih_diatomic.yml": """
system:
  module: diatomic
  formula: LiH
  bond_length: 3.015
  unit: bohr
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
        yaml_files=("lih_diatomic.yml",),
    ),
    CliDryRunCase(
        "molecule_alkane_yaml",
        "molecule",
        "train",
        files={
            "ethane_alkane.yml": """
system:
  module: alkane
  repeat_num: 1
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
        yaml_files=("ethane_alkane.yml",),
    ),
    CliDryRunCase(
        "molecule_ecp_atom_yaml",
        "molecule",
        "train",
        files={
            "fe_ecp.yml": """
system:
  module: atom
  symbol: Fe
  pp: ccecp
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
        yaml_files=("fe_ecp.yml",),
    ),
    CliDryRunCase(
        "molecule_psiformer_dotlist",
        "molecule",
        "train",
        dotlist=(
            "wf.module=psiformer",
            "workflow.batch_size=4",
            "wf.num_layers=1",
            "wf.num_heads=1",
            "wf.heads_dim=4",
            "wf.mlp_hidden_dims=[4]",
            "pretrain.run.iterations=1",
            "train.run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "molecule_lapnet_dotlist",
        "molecule",
        "train",
        dotlist=(
            "wf.module=lapnet",
            "workflow.batch_size=4",
            "wf.num_layers=1",
            "wf.num_heads=2",
            "wf.heads_dim=8",
            "wf.ndets=4",
            "pretrain.run.iterations=1",
            "train.run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "solid_two_atom_chain_yaml",
        "solid",
        "train",
        files={
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
        yaml_files=("solid.yml",),
    ),
    CliDryRunCase(
        "solid_arbitrary_lih_yaml",
        "solid",
        "train",
        files={
            "lih_solid.yml": """
system:
  lattice:
    a: [0.0, 3.78, 3.78]
    b: [3.78, 0.0, 3.78]
    c: [3.78, 3.78, 0.0]
  atoms:
    - symbol: Li
      frac_coords: [0.0, 0.0, 0.0]
    - symbol: H
      frac_coords: [0.5, 0.5, 0.5]
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
        yaml_files=("lih_solid.yml",),
    ),
    CliDryRunCase(
        "solid_rock_salt_yaml",
        "solid",
        "train",
        files={
            "rock_salt.yml": """
system:
  module: rock_salt
  symbol_a: Li
  symbol_b: H
  lattice_constant: 4.0
  unit: angstrom
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
        yaml_files=("rock_salt.yml",),
    ),
    CliDryRunCase(
        "electron_gas_train_dotlist",
        "electron-gas",
        "train",
        dotlist=(
            "system.rs=1",
            "system.nelectrons=2",
            "system.s_z=0",
            "workflow.batch_size=4",
            "wf.hidden_dims_single=[4]",
            "wf.hidden_dims_double=[2]",
            "wf.ndets=1",
            "pretrain.run.iterations=1",
            "train.run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "electron_gas_evaluate_dotlist",
        "electron-gas",
        "evaluate",
        dotlist=(
            "system.rs=1",
            "system.nelectrons=2",
            "system.s_z=0",
            "workflow.batch_size=4",
            "workflow.source_path=source",
            "wf.hidden_dims_single=[4]",
            "wf.hidden_dims_double=[2]",
            "wf.ndets=1",
            "run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "hall_train_yaml",
        "hall",
        "train",
        files={
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
        yaml_files=("hall.yml",),
    ),
    CliDryRunCase(
        "hall_train_penalty_dotlist",
        "hall",
        "train",
        dotlist=(
            "system.lz_penalty=10",
            "system.lz_center=0",
            "workflow.batch_size=4",
            "train.run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "hall_train_composite_fermion_dotlist",
        "hall",
        "train",
        dotlist=(
            "system.flux=10",
            "system.nspins=[4,0]",
            "wf.flux_per_elec=2",
            "workflow.batch_size=4",
            "train.run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "hall_train_laughlin_module_dotlist",
        "hall",
        "train",
        dotlist=(
            "wf.module=laughlin",
            "system.flux=10",
            "system.nspins=[4,0]",
            "workflow.batch_size=4",
            "train.run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "moire_train_dotlist",
        "moire",
        "train",
        dotlist=("workflow.batch_size=4", "train.run.iterations=1"),
    ),
    CliDryRunCase(
        "molecule_evaluate_yaml",
        "molecule",
        "evaluate",
        files={
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
        yaml_files=("eval.yml",),
    ),
    CliDryRunCase(
        "solid_evaluate_density_yaml",
        "solid",
        "evaluate",
        files={
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
        yaml_files=("solid_eval.yml",),
    ),
    CliDryRunCase(
        "hall_evaluate_observables_dotlist",
        "hall",
        "evaluate",
        dotlist=(
            "workflow.batch_size=4",
            "workflow.source_path=source",
            "run.iterations=1",
            "estimators.enabled.density=true",
            "estimators.enabled.pair_correlation=true",
            "estimators.enabled.one_rdm=true",
        ),
    ),
    CliDryRunCase(
        "hall_evaluate_laughlin_no_source",
        "hall",
        "evaluate",
        dotlist=(
            "system.flux=6",
            "system.nspins=[3,0]",
            "wf.module=laughlin",
            "workflow.batch_size=4",
            "run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "hall_evaluate_free_no_source",
        "hall",
        "evaluate",
        dotlist=(
            "system.flux=6",
            "system.nspins=[3,0]",
            "wf.module=free",
            "workflow.batch_size=4",
            "run.iterations=1",
        ),
    ),
    CliDryRunCase(
        "moire_evaluate_dotlist",
        "moire",
        "evaluate",
        dotlist=(
            "workflow.batch_size=4",
            "workflow.source_path=source",
            "run.iterations=1",
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_cli_command_dry_run(
    case: CliDryRunCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name, text in case.files.items():
        (tmp_path / name).write_text(text.strip() + "\n", encoding="utf8")
    argv = [*case.argv, "--dry-run"]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, f"command: {argv}\noutput:\n{result.output}"
    written = [
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.name not in case.files
    ]
    assert not written, f"dry-run wrote files: {written}"


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


def test_cli_invalid_dotlist_shows_click_error() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "hall",
            "train",
            "--dry-run",
            "workflow.batch_size",
            "train.run.iterations=1",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid CLI override 'workflow.batch_size'" in result.output


def test_cli_unused_key_lists_exact_path() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "hall",
            "train",
            "--dry-run",
            "definitely_not_exist.batch_size=4",
            "train.run.iterations=1",
        ],
    )

    assert result.exit_code == 1
    assert "Unused config keys detected" in result.output
    assert "definitely_not_exist.batch_size" in result.output


def test_cli_yaml_root_type_error(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf8")

    result = CliRunner().invoke(
        cli,
        ["hall", "train", "--yml", str(config_path), "--dry-run"],
    )

    assert result.exit_code == 1
    assert f"Invalid YAML config in '{config_path}'" in result.output


def test_cli_unknown_field_uses_wrapped_pyserde_message() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "hall",
            "evaluate",
            "--dry-run",
            "workflow.batch_size=4",
            "workflow.source_path=source",
            "workflow.source_pat=source",
            "run.iterations=1",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid config at 'workflow': unknown fields:" in result.output
