# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
from click.testing import CliRunner
from upath import UPath

from jaqmc.app.cli import cli
from jaqmc.app.solid.reference import pyscf as reference_pyscf
from jaqmc.app.solid.reference import qe


@pytest.mark.parametrize(
    ("input_kind", "expected_source"),
    [("pyscf", "pyscf"), ("qe", "qe")],
)
@pytest.mark.parametrize(
    "output_uri",
    ["{tmp_path}/reference.npz", "memory://jaqmc-solid-reference/reference.npz"],
)
def test_solid_convert_cli_infers_source(
    tmp_path, monkeypatch, input_kind, expected_source, output_uri
):
    if input_kind == "pyscf":
        input_path = tmp_path / "solver.chk"
        input_path.touch()
    else:
        input_path = tmp_path / "solver.save"
        input_path.mkdir()
        (input_path / "data-file-schema.xml").touch()
    output_path = UPath(output_uri.format(tmp_path=tmp_path))
    converted: list[tuple[str, object, object]] = []
    monkeypatch.setattr(
        reference_pyscf,
        "convert_checkpoint",
        lambda source, target: converted.append(("pyscf", source, target)),
    )
    monkeypatch.setattr(
        qe,
        "convert_save_directory",
        lambda source, target: converted.append(("qe", source, target)),
    )

    result = CliRunner().invoke(
        cli,
        [
            "solid",
            "reference",
            "convert",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert converted == [(expected_source, input_path, output_path)]
