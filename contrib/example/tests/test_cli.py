# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from click.testing import CliRunner
from jaqmc_contrib_example.cli import cli


def test_example_contrib_info() -> None:
    result = CliRunner().invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "jaqmc-contrib-example" in result.output
    assert "configured package: jaqmc-contrib-example" in result.output
    assert "jaqmc version:" in result.output


def test_example_contrib_version() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "jaqmc version:" in result.output
