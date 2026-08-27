# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy discovery of jaqmc.apps CLI entry points."""

from collections.abc import Generator
from importlib.metadata import EntryPoint
from types import ModuleType
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from jaqmc.app.cli import ENTRY_POINT_GROUP, cli


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


def _patch_entry_points(entry_point_list: list[EntryPoint]) -> patch:
    return patch(
        "jaqmc.app.cli.entry_points",
        return_value=entry_point_list,
    )


@pytest.fixture(autouse=True)
def _clear_contrib_entry_point_cache() -> Generator[None, None, None]:
    cli.__dict__.pop("_contributed_entry_points", None)
    yield
    cli.__dict__.pop("_contributed_entry_points", None)


def test_builtin_commands_remain_available() -> None:
    with _patch_entry_points([]) as mocked_entry_points:
        result = CliRunner().invoke(cli, ["molecule", "--help"])

    assert result.exit_code == 0
    assert "Molecular system workflows." in result.output
    mocked_entry_points.assert_not_called()


def test_contrib_command_loads_on_demand() -> None:
    plugin = ModuleType("test_contrib_cli_plugin")

    @click.command()
    def contrib_cli() -> None:
        click.echo("loaded contributed command")

    plugin.__dict__["contrib_cli"] = contrib_cli
    with (
        _patch_entry_points(
            [_entry_point("demo-contrib", "test_contrib_cli_plugin:contrib_cli")]
        ),
        patch.dict("sys.modules", {plugin.__name__: plugin}),
    ):
        result = CliRunner().invoke(cli, ["demo-contrib"])

    assert result.exit_code == 0
    assert "loaded contributed command" in result.output


def test_duplicate_entry_point_names_rejected() -> None:
    with _patch_entry_points(
        [
            _entry_point("dup-contrib", "jaqmc_contrib_example.cli:cli"),
            _entry_point("dup-contrib", "jaqmc_contrib_example.cli:cli"),
        ]
    ):
        result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code != 0
    assert "duplicate jaqmc.apps providers" in result.output
