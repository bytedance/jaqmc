# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from click.testing import CliRunner
from jaqmc_contrib_lit.cli import lit

from jaqmc.app.cli import cli as jaqmc_cli


def test_lit_plugin_exposes_only_run_and_invert_commands():
    assert set(lit.commands) == {"run", "invert"}
    result = CliRunner().invoke(lit, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "invert" in result.output


def test_he_example_configs_resolve_through_public_cli(caplog):
    config_dir = Path(__file__).parents[1] / "examples" / "he" / "config"
    runner = CliRunner()

    ground = runner.invoke(
        jaqmc_cli,
        ["molecule", "train", "--yml", str(config_dir / "ground.yml"), "--dry-run"],
    )
    spectrum = runner.invoke(
        jaqmc_cli,
        ["lit", "run", "--yml", str(config_dir / "lit.yml"), "--dry-run"],
    )
    inversion = runner.invoke(
        jaqmc_cli,
        ["lit", "invert", "--yml", str(config_dir / "invert.yml"), "--dry-run"],
    )

    assert ground.exit_code == 0, ground.output
    assert spectrum.exit_code == 0, spectrum.output
    assert inversion.exit_code == 0, inversion.output
    assert "minimum: 0.75" in caplog.text
    assert "train_batch_size_per_device: 1024" in caplog.text
    assert "pole_count: 1" in caplog.text


@pytest.mark.parametrize("pseudopotential", ["ccecp", "ph"])
def test_lit_rejects_pseudopotential_hamiltonians(pseudopotential):
    result = CliRunner().invoke(
        jaqmc_cli,
        [
            "lit",
            "run",
            "system.module=atom",
            "system.symbol=Fe",
            f"system.pp={pseudopotential}",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "supports only all-electron molecular Hamiltonians" in str(result.exception)
    assert f"Fe={pseudopotential}" in str(result.exception)
