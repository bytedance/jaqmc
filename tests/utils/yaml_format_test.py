# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from jaqmc.utils.yaml_format import dump_yaml, load_yaml


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1e-6", 1e-6), ("1.0e6", 1e6), ("-2E+3", -2e3), (".5e1", 5.0), ("1.5", 1.5)],
)
def test_load_yaml_reads_exponent_floats(text, expected):
    assert load_yaml(f"x: {text}") == {"x": pytest.approx(expected)}


@pytest.mark.parametrize(
    ("text", "expected"), [("1_000", 1000), ("e5", "e5"), ("1.2.3", "1.2.3")]
)
def test_load_yaml_keeps_other_scalars(text, expected):
    assert load_yaml(f"x: {text}") == {"x": expected}


class TestDumpYaml:
    def test_numeric_list_flow_style(self):
        data = {"coords": [1.0, 2.0, 3.0]}
        yaml_str = dump_yaml(data)
        assert "coords: [1.0, 2.0, 3.0]" in yaml_str

    def test_matrix_flow_style_rows(self):
        data = {
            "lattice": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        }
        yaml_str = dump_yaml(data)
        assert "- [1.0, 0.0, 0.0]" in yaml_str
        assert "- [0.0, 1.0, 0.0]" in yaml_str
        assert "- [0.0, 0.0, 1.0]" in yaml_str

    def test_mixed_list_block_style(self):
        data = {"mixed": [1, "text", 3]}
        yaml_str = dump_yaml(data)
        # Mixed lists should use block style
        assert "- 1" in yaml_str
        assert "- text" in yaml_str
        assert "- 3" in yaml_str

    def test_empty_list(self):
        data: dict[str, list[float]] = {"empty": []}
        yaml_str = dump_yaml(data)
        assert "empty: []" in yaml_str

    def test_accepts_plain_dict(self):
        data = {"coords": [1.0, 2.0, 3.0], "name": "test"}
        yaml_str = dump_yaml(data)
        assert "coords: [1.0, 2.0, 3.0]" in yaml_str
        assert "name: test" in yaml_str

    def test_roundtrip_via_yaml(self):
        original = {
            "name": "true",
            "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "coords": [1.0, 2.0, 3.0],
            "nested": {"value": 42},
        }
        yaml_str = dump_yaml(original)
        loaded = yaml.safe_load(yaml_str)

        assert loaded["name"] == "true"
        assert loaded["coords"] == [1.0, 2.0, 3.0]
        assert loaded["lattice"] == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert loaded["nested"]["value"] == 42

    def test_sort_keys(self):
        data = {"z": 1, "a": 2, "m": 3}
        yaml_str = dump_yaml(data, sort_keys=True)
        lines = [line for line in yaml_str.splitlines() if line.strip()]
        assert lines[0].startswith("a:")
        assert lines[1].startswith("m:")
        assert lines[2].startswith("z:")

    def test_tuple_serialized_as_list(self):
        data = {"spins": (3, 2)}
        yaml_str = dump_yaml(data)
        assert "spins: [3, 2]" in yaml_str
